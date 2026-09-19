using System.Collections.Concurrent;

namespace MACE.Online
{
    /// <summary>Configuration for an <see cref="InferencePool"/>.</summary>
    /// <param name="NumWorkers">Number of workers the pooled models know about.</param>
    /// <param name="NumCategories">Number of label categories.</param>
    /// <param name="PoolSize">How many models to build, which caps concurrent inference.</param>
    /// <param name="Iterations">Number of EP inference iterations per call.</param>
    /// <param name="Seed">
    /// Optional RNG seed. Every model in the pool is built with it, so which slot a request lands on
    /// does not change the answer.
    /// </param>
    public record InferencePoolOptions(
        int NumWorkers,
        int NumCategories,
        int PoolSize = 4,
        int Iterations = 50,
        int? Seed = null);

    /// <summary>A pool of ready-to-use single-item models.</summary>
    public interface IInferencePool
    {
        /// <summary>Takes exclusive use of a model, waiting if all are busy.</summary>
        /// <param name="cancellationToken">Abandons the wait.</param>
        /// <returns>A lease that must be disposed to return the model.</returns>
        Task<PooledInference> AcquireAsync(CancellationToken cancellationToken = default);

        /// <summary>Models not currently leased.</summary>
        int Available { get; }

        /// <summary>Total models in the pool.</summary>
        int Total { get; }
    }

    /// <summary>
    /// Holds a fixed set of <see cref="MACETrain"/> instances and lends them out one caller at a time.
    /// </summary>
    /// <remarks>
    /// Infer.NET's <c>InferenceEngine</c> is not thread-safe — setting observed values and running
    /// inference both mutate engine state — so concurrent callers cannot share one model. Building a
    /// model per call is not the answer either: the first inference on a fresh model compiles the
    /// factor graph, which costs seconds. The pool builds its models once and hands out exclusive
    /// leases, which bounds concurrency to <see cref="Total"/> and keeps compilation off the request
    /// path.
    ///
    /// <para>
    /// The models are warmed during construction rather than on first use, so that cost lands at
    /// startup where it can be waited on, instead of inside whichever request arrives first.
    /// </para>
    /// </remarks>
    public sealed class InferencePool : IInferencePool, IDisposable
    {
        private readonly SemaphoreSlim _slots;
        private readonly ConcurrentQueue<MACETrain> _available;
        private readonly int _total;
        private bool _disposed;

        /// <summary>Builds and warms every model in the pool.</summary>
        /// <param name="options">Pool configuration.</param>
        /// <param name="onProgress">Optional callback for warm-up messages, one per model.</param>
        /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the pool size is not positive.</exception>
        public InferencePool(InferencePoolOptions options, Action<string>? onProgress = null)
        {
            ArgumentNullException.ThrowIfNull(options);

            if (options.PoolSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(options), "Pool size must be positive.");

            _total = options.PoolSize;
            _slots = new SemaphoreSlim(_total, _total);
            _available = new ConcurrentQueue<MACETrain>();

            var warmUpAnnotations = WarmUpAnnotations(options.NumWorkers, options.NumCategories);
            var warmUpPriors = UniformPriors(options.NumWorkers, options.NumCategories);

            for (int slot = 0; slot < _total; slot++)
            {
                var model = MACETrain.ForOnlineInference(
                    options.NumWorkers, options.NumCategories, options.Iterations, options.Seed);

                // Infer.NET writes iteration progress straight to the console, which would bypass
                // whatever logger the host has configured and emit a line per request.
                model.ShowProgress = false;

                // Run one throwaway inference so this model has compiled before it serves a request.
                model.InferOnline(warmUpAnnotations, warmUpPriors);

                _available.Enqueue(model);
                onProgress?.Invoke($"Inference slot {slot + 1}/{_total} ready.");
            }
        }

        /// <inheritdoc />
        public int Available => _available.Count;

        /// <inheritdoc />
        public int Total => _total;

        /// <inheritdoc />
        public async Task<PooledInference> AcquireAsync(CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            await _slots.WaitAsync(cancellationToken).ConfigureAwait(false);

            if (_available.TryDequeue(out var model))
            {
                return new PooledInference(model, () => Release(model));
            }

            // The semaphore count and the queue length are maintained together, so this cannot
            // happen; releasing the permit keeps the pool usable if it somehow does.
            _slots.Release();
            throw new InvalidOperationException("Inference pool is inconsistent: a slot was granted but no model was free.");
        }

        private void Release(MACETrain model)
        {
            _available.Enqueue(model);
            _slots.Release();
        }

        /// <summary>Releases the pool's semaphore. Leases taken out beforehand remain usable.</summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _slots.Dispose();
        }

        /// <summary>
        /// A single annotation is enough to compile the model, and is the cheapest input that
        /// exercises the observed path.
        /// </summary>
        private static int[] WarmUpAnnotations(int numWorkers, int numCategories)
        {
            var annotations = Enumerable.Repeat(MACETrain.MissingAnnotation, numWorkers).ToArray();
            if (numWorkers > 0 && numCategories > 0)
            {
                annotations[0] = 0;
            }

            return annotations;
        }

        private static ModelPriors UniformPriors(int numWorkers, int numCategories)
        {
            var concentration = Enumerable.Repeat(1.0, numCategories).ToArray();
            return new ModelPriors(
                ThetaDist: Enumerable.Range(0, numWorkers)
                    .Select(_ => new Microsoft.ML.Probabilistic.Distributions.Beta(1, 1)).ToArray(),
                PhiDist: Enumerable.Range(0, numWorkers)
                    .Select(_ => new Microsoft.ML.Probabilistic.Distributions.Dirichlet(concentration)).ToArray());
        }
    }

    /// <summary>
    /// Exclusive use of a pooled model, returned to the pool when disposed.
    /// </summary>
    /// <remarks>
    /// The release callback keeps this decoupled from the pool type, so tests can build a lease over
    /// a model they own with a no-op release.
    /// </remarks>
    public readonly struct PooledInference : IDisposable
    {
        private readonly Action _release;

        /// <summary>Creates a lease over a model.</summary>
        /// <param name="model">The leased model.</param>
        /// <param name="release">Called once when the lease is disposed.</param>
        public PooledInference(MACETrain model, Action release)
        {
            Model = model;
            _release = release;
        }

        /// <summary>The leased model. Valid until this lease is disposed.</summary>
        public MACETrain Model { get; }

        /// <summary>Returns the model to the pool.</summary>
        public void Dispose() => _release();
    }
}
