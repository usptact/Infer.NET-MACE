using Grpc.Core;
using MACE;
using MACE.Online;
using MACE.Service;
using MACE.Service.Grpc;
using MACE.Service.Services;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Microsoft.ML.Probabilistic.Distributions;

namespace MACE.Tests.Online
{
    /// <summary>
    /// A server call context with no transport behind it, so the service can be exercised without
    /// binding a port.
    /// </summary>
    internal sealed class TestCallContext : ServerCallContext
    {
        private readonly CancellationToken _token;

        public TestCallContext(CancellationToken token = default) => _token = token;

        protected override string MethodCore => "/mace.inference.v1.MaceInference/Test";
        protected override string HostCore => "localhost";
        protected override string PeerCore => "test";
        protected override DateTime DeadlineCore => DateTime.MaxValue;
        protected override Metadata RequestHeadersCore => new();
        protected override CancellationToken CancellationTokenCore => _token;
        protected override Metadata ResponseTrailersCore => new();
        protected override Status StatusCore { get; set; }
        protected override WriteOptions? WriteOptionsCore { get; set; }
        protected override AuthContext AuthContextCore => new(null, new Dictionary<string, List<AuthProperty>>());

        protected override ContextPropagationToken CreatePropagationTokenCore(ContextPropagationOptions? options)
            => throw new NotSupportedException();

        protected override Task WriteResponseHeadersAsyncCore(Metadata responseHeaders) => Task.CompletedTask;
    }

    public class MaceInferenceServiceTests : IDisposable
    {
        private const int Workers = 4;
        private const int Categories = 3;

        private readonly InferencePool _pool;
        private readonly BeliefStore _beliefs;
        private readonly MaceInferenceService _service;

        public MaceInferenceServiceTests()
        {
            var options = new ServiceOptions
            {
                NumWorkers = Workers,
                NumCategories = Categories,
                PoolSize = 2,
                Iterations = 50,
                Seed = 42,
                LearningRate = 1.0,
                Retention = 1.0
            };

            _pool = new InferencePool(
                new InferencePoolOptions(Workers, Categories, options.PoolSize, options.Iterations, options.Seed));

            var concentration = Enumerable.Repeat(1.0, Categories).ToArray();
            _beliefs = new BeliefStore(
                new ModelPriors(
                    Enumerable.Range(0, Workers).Select(_ => new Beta(1, 1)).ToArray(),
                    Enumerable.Range(0, Workers).Select(_ => new Dirichlet(concentration)).ToArray()),
                new PriorUpdateService());

            _service = new MaceInferenceService(
                _pool, _beliefs, Options.Create(options), NullLogger<MaceInferenceService>.Instance);
        }

        public void Dispose() => _pool.Dispose();

        private static InferLabelRequest Request(string itemId, params (int Worker, int Label)[] annotations)
        {
            var request = new InferLabelRequest { ItemId = itemId };
            foreach (var (worker, label) in annotations)
            {
                request.Annotations.Add(new Annotation { Worker = worker, Label = label });
            }

            return request;
        }

        [Fact]
        public async Task InferLabel_ReturnsTheAgreedLabelAndEchoesTheItemId()
        {
            var response = await _service.InferLabel(
                Request("item-1", (0, 1), (1, 1), (2, 1)), new TestCallContext());

            Assert.Equal("item-1", response.ItemId);
            Assert.Equal(1, response.Label);
            Assert.True(response.Confidence > 0.9);
            Assert.Equal(Categories, response.LabelProbabilities.Count);
            Assert.Equal(new[] { 0, 1, 2 }, response.ContributingWorkers);
        }

        [Fact]
        public async Task InferLabel_ReportsOnlyTheWorkersWhoAnnotated()
        {
            var response = await _service.InferLabel(
                Request("item-2", (0, 0), (3, 0)), new TestCallContext());

            Assert.Equal(2, response.WorkerAssessments.Count);
            Assert.Equal(new[] { 0, 3 }, response.WorkerAssessments.Select(a => a.Worker));
            Assert.All(response.WorkerAssessments, a => Assert.InRange(a.SpammerProbability, 0.0, 1.0));
        }

        [Fact]
        public async Task InferLabel_WithNoAnnotations_ReturnsThePriorAndSaysItHasNoEvidence()
        {
            var response = await _service.InferLabel(Request("item-3"), new TestCallContext());

            Assert.Empty(response.ContributingWorkers);
            Assert.Empty(response.WorkerAssessments);
            Assert.Equal(Math.Log(Categories), response.Entropy, precision: 6);
        }

        [Fact]
        public async Task InferLabel_DoesNotChangeWorkerReliability()
        {
            var before = _beliefs.Snapshot();
            await _service.InferLabel(Request("item-4", (0, 0), (1, 1)), new TestCallContext());
            var after = _beliefs.Snapshot();

            Assert.Same(before, after);
        }

        /// <summary>
        /// A worker sent twice in one request is a caller mistake. Keeping the last value would make
        /// the answer depend on request ordering, so it is rejected instead.
        /// </summary>
        [Fact]
        public async Task InferLabel_DuplicateWorker_IsRejected()
        {
            var ex = await Assert.ThrowsAsync<RpcException>(
                () => _service.InferLabel(Request("item-5", (1, 0), (1, 2)), new TestCallContext()));

            Assert.Equal(StatusCode.InvalidArgument, ex.StatusCode);
            Assert.Contains("more than once", ex.Status.Detail);
        }

        [Theory]
        [InlineData(9, 0)]
        [InlineData(-1, 0)]
        public async Task InferLabel_WorkerOutOfRange_IsRejected(int worker, int label)
        {
            var ex = await Assert.ThrowsAsync<RpcException>(
                () => _service.InferLabel(Request("item-6", (worker, label)), new TestCallContext()));

            Assert.Equal(StatusCode.InvalidArgument, ex.StatusCode);
            Assert.Contains("Worker", ex.Status.Detail);
        }

        [Fact]
        public async Task InferLabel_LabelOutOfRange_IsRejected()
        {
            var ex = await Assert.ThrowsAsync<RpcException>(
                () => _service.InferLabel(Request("item-7", (0, 7)), new TestCallContext()));

            Assert.Equal(StatusCode.InvalidArgument, ex.StatusCode);
            Assert.Contains("Label 7", ex.Status.Detail);
        }

        [Fact]
        public async Task SubmitFeedback_MovesOnlyTheWorkersWhoAnnotated()
        {
            var request = new SubmitFeedbackRequest { ItemId = "item-8", TrueLabel = 0 };
            request.Annotations.Add(new Annotation { Worker = 0, Label = 0 });
            request.Annotations.Add(new Annotation { Worker = 1, Label = 2 });

            var response = await _service.SubmitFeedback(request, new TestCallContext());

            Assert.Equal("item-8", response.ItemId);
            Assert.Equal(new[] { 0, 1 }, response.Updated.Select(u => u.Worker));

            // Worker 1 contradicted the true label, so they must look worse than worker 0.
            var agreed = response.Updated.Single(u => u.Worker == 0);
            var disagreed = response.Updated.Single(u => u.Worker == 1);
            Assert.True(disagreed.SpammerProbability > agreed.SpammerProbability);

            // Each carries the evidence it has earned above the uninformative prior.
            Assert.True(disagreed.Evidence > 0.0);
        }

        [Fact]
        public async Task SubmitFeedback_TrueLabelOutOfRange_IsRejected()
        {
            var request = new SubmitFeedbackRequest { ItemId = "item-9", TrueLabel = 5 };

            var ex = await Assert.ThrowsAsync<RpcException>(
                () => _service.SubmitFeedback(request, new TestCallContext()));

            Assert.Equal(StatusCode.InvalidArgument, ex.StatusCode);
            Assert.Contains("true_label 5", ex.Status.Detail);
        }

        [Theory]
        [InlineData(0.0, 0.5)]
        [InlineData(0.5, 0.0)]
        [InlineData(0.5, 1.5)]
        public async Task SubmitFeedback_InvalidRates_AreRejected(double learningRate, double retention)
        {
            var request = new SubmitFeedbackRequest
            {
                ItemId = "item-10",
                TrueLabel = 0,
                LearningRate = learningRate,
                Retention = retention
            };

            var ex = await Assert.ThrowsAsync<RpcException>(
                () => _service.SubmitFeedback(request, new TestCallContext()));

            Assert.Equal(StatusCode.InvalidArgument, ex.StatusCode);
        }

        [Fact]
        public async Task GetWorkerReliability_ReturnsEveryWorkerByDefault()
        {
            var response = await _service.GetWorkerReliability(
                new GetWorkerReliabilityRequest(), new TestCallContext());

            Assert.Equal(Workers, response.Workers.Count);
            Assert.All(response.Workers, w => Assert.Equal(Categories, w.SpamPreferences.Count));
            Assert.All(response.Workers, w => Assert.Equal(0.5, w.SpammerProbability, precision: 9));
            Assert.All(response.Workers, w => Assert.Equal(0.0, w.Evidence, precision: 9));
        }

        [Fact]
        public async Task GetWorkerReliability_CanSelectSpecificWorkers()
        {
            var request = new GetWorkerReliabilityRequest();
            request.Workers.Add(2);

            var response = await _service.GetWorkerReliability(request, new TestCallContext());

            Assert.Single(response.Workers);
            Assert.Equal(2, response.Workers[0].Worker);
        }

        [Fact]
        public async Task GetWorkerReliability_UnknownWorker_IsRejected()
        {
            var request = new GetWorkerReliabilityRequest();
            request.Workers.Add(99);

            var ex = await Assert.ThrowsAsync<RpcException>(
                () => _service.GetWorkerReliability(request, new TestCallContext()));

            Assert.Equal(StatusCode.InvalidArgument, ex.StatusCode);
        }

        /// <summary>
        /// The loop the service exists for: feedback should change what later inference concludes.
        /// After a worker is repeatedly shown to be wrong, their dissent should carry less weight.
        /// </summary>
        [Fact]
        public async Task FeedbackChangesWhatLaterInferenceConcludes()
        {
            var context = new TestCallContext();

            // Worker 3 disagrees with everyone, every time, and is repeatedly shown to be wrong.
            for (int round = 0; round < 25; round++)
            {
                int truth = round % Categories;
                int wrong = (truth + 1) % Categories;

                var feedback = new SubmitFeedbackRequest { ItemId = $"train-{round}", TrueLabel = truth };
                feedback.Annotations.Add(new Annotation { Worker = 0, Label = truth });
                feedback.Annotations.Add(new Annotation { Worker = 1, Label = truth });
                feedback.Annotations.Add(new Annotation { Worker = 3, Label = wrong });

                await _service.SubmitFeedback(feedback, context);
            }

            var reliability = await _service.GetWorkerReliability(new GetWorkerReliabilityRequest(), context);
            double discredited = reliability.Workers.Single(w => w.Worker == 3).SpammerProbability;
            Assert.True(discredited > 0.8, $"Worker 3 should read as unreliable, got {discredited}.");

            // Now a single trusted worker against the discredited one: the trusted worker should win.
            var response = await _service.InferLabel(Request("live", (0, 2), (3, 0)), context);

            Assert.Equal(2, response.Label);
            Assert.True(
                response.WorkerAssessments.Single(a => a.Worker == 3).SpammerProbability >
                response.WorkerAssessments.Single(a => a.Worker == 0).SpammerProbability);
        }
    }
}
