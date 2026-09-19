using MACE;
using MACE.Online;

namespace MACE.Tests.Online
{
    public class InferencePoolTests
    {
        private const int Workers = 4;
        private const int Categories = 2;

        private static InferencePoolOptions Options(int poolSize = 2)
            => new(Workers, Categories, PoolSize: poolSize, Iterations: 50, Seed: 42);

        [Fact]
        public void PoolReportsItsSizeAndStartsFullyAvailable()
        {
            using var pool = new InferencePool(Options(poolSize: 3));

            Assert.Equal(3, pool.Total);
            Assert.Equal(3, pool.Available);
        }

        [Fact]
        public void WarmUpProgressIsReportedPerSlot()
        {
            var messages = new List<string>();
            using var pool = new InferencePool(Options(poolSize: 2), messages.Add);

            Assert.Equal(2, messages.Count);
            Assert.Contains("1/2", messages[0]);
            Assert.Contains("2/2", messages[1]);
        }

        [Fact]
        public async Task AcquiringTakesASlotAndDisposingReturnsIt()
        {
            using var pool = new InferencePool(Options(poolSize: 2));

            using (var lease = await pool.AcquireAsync())
            {
                Assert.Equal(1, pool.Available);
                Assert.NotNull(lease.Model);
            }

            Assert.Equal(2, pool.Available);
        }

        [Fact]
        public async Task ALeasedModelCanRunInference()
        {
            using var pool = new InferencePool(Options());
            using var lease = await pool.AcquireAsync();

            var result = lease.Model.InferOnline(
                new[] { 1, 1, 1, 1 },
                TestSupport.UniformPriors(Workers, Categories));

            Assert.Equal(1, result.Label);
            Assert.True(result.Confidence > 0.9);
        }

        /// <summary>
        /// Two callers must never hold the same model: Infer.NET's engine is not thread-safe, so
        /// sharing one would interleave observed values between requests.
        /// </summary>
        [Fact]
        public async Task ConcurrentLeasesHandOutDistinctModels()
        {
            using var pool = new InferencePool(Options(poolSize: 2));

            using var first = await pool.AcquireAsync();
            using var second = await pool.AcquireAsync();

            Assert.NotSame(first.Model, second.Model);
            Assert.Equal(0, pool.Available);
        }

        /// <summary>
        /// Once every model is out, the next caller waits rather than being handed a shared model or
        /// an error.
        /// </summary>
        [Fact]
        public async Task AcquiringAnExhaustedPoolWaits()
        {
            using var pool = new InferencePool(Options(poolSize: 1));
            using var held = await pool.AcquireAsync();

            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(250));

            await Assert.ThrowsAnyAsync<OperationCanceledException>(
                async () => await pool.AcquireAsync(cts.Token));
        }

        [Fact]
        public async Task AWaitingCallerProceedsOnceASlotIsReturned()
        {
            using var pool = new InferencePool(Options(poolSize: 1));

            var lease = await pool.AcquireAsync();
            var waiting = pool.AcquireAsync();

            Assert.False(waiting.IsCompleted);

            lease.Dispose();

            using var acquired = await waiting.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.NotNull(acquired.Model);
        }

        /// <summary>
        /// Every model is built with the same seed, so which slot a request happens to land on must
        /// not change the answer.
        /// </summary>
        [Fact]
        public async Task EverySlotProducesTheSameAnswer()
        {
            using var pool = new InferencePool(Options(poolSize: 3));
            var priors = TestSupport.UniformPriors(Workers, Categories);
            var annotations = new[] { 0, 1, 0, 0 };

            var leases = new List<PooledInference>();
            var answers = new List<double>();

            for (int slot = 0; slot < pool.Total; slot++)
            {
                var lease = await pool.AcquireAsync();
                leases.Add(lease);
                answers.Add(lease.Model.InferOnline(annotations, priors).LabelDist.GetProbs()[0]);
            }

            foreach (var lease in leases)
            {
                lease.Dispose();
            }

            foreach (var answer in answers)
            {
                Assert.Equal(answers[0], answer, precision: 12);
            }
        }

        [Fact]
        public async Task ParallelCallersAllCompleteAndAgree()
        {
            using var pool = new InferencePool(Options(poolSize: 3));
            var priors = TestSupport.UniformPriors(Workers, Categories);

            var tasks = Enumerable.Range(0, 12).Select(async _ =>
            {
                using var lease = await pool.AcquireAsync();
                return lease.Model.InferOnline(new[] { 1, 1, 1, 0 }, priors).Label;
            });

            var labels = await Task.WhenAll(tasks);

            Assert.Equal(12, labels.Length);
            Assert.Single(labels.Distinct());
            Assert.Equal(3, pool.Available);
        }

        [Fact]
        public void NonPositivePoolSize_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new InferencePool(Options(poolSize: 0)));
        }

        [Fact]
        public void NullOptions_Throw()
        {
            Assert.Throws<ArgumentNullException>(() => new InferencePool(null!));
        }

        [Fact]
        public async Task AcquiringFromADisposedPool_Throws()
        {
            var pool = new InferencePool(Options(poolSize: 1));
            pool.Dispose();

            await Assert.ThrowsAsync<ObjectDisposedException>(async () => await pool.AcquireAsync());
        }

        [Fact]
        public void DisposingTwiceIsSafe()
        {
            var pool = new InferencePool(Options(poolSize: 1));
            pool.Dispose();
            pool.Dispose();
        }
    }
}
