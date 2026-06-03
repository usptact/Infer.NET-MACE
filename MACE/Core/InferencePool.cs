using System.Collections.Concurrent;
using Microsoft.Extensions.Options;

namespace MACE.Core;

/// <summary>
/// Thread-safe pool of pre-warmed <see cref="MACETrain"/> instances.
///
/// Infer.NET's InferenceEngine is not thread-safe: Infer&lt;T&gt;(), ObservedValue
/// assignment, and InitialiseTo() all mutate internal state. Each concurrent
/// HTTP request must own its instance exclusively.
///
/// Creating a new MACETrain per request is too expensive because CreateModel()
/// compiles the Infer.NET factor graph (~1–3 s). The pool pays that cost once
/// per slot at startup and leases slots to requests.
/// </summary>
public sealed class InferencePool : IDisposable
{
    private readonly SemaphoreSlim _semaphore;
    private readonly ConcurrentQueue<MACETrain> _available;
    private readonly int _total;
    private bool _disposed;

    public InferencePool(IOptions<InferenceOptions> options, ILogger<InferencePool> logger)
    {
        var opts = options.Value;
        _total     = opts.PoolSize;
        _semaphore = new SemaphoreSlim(_total, _total);
        _available = new ConcurrentQueue<MACETrain>();

        logger.LogInformation(
            "Initialising inference pool ({Size} slots, {Sensors} sensor types, {Cats} categories)...",
            _total, opts.NumSensorTypes, opts.NumCategories);

        for (int i = 0; i < _total; i++)
        {
            // Online constructor: numItems is always 1
            var trainer = new MACETrain(opts.NumSensorTypes, opts.NumCategories);
            trainer.CreateModel();
            _available.Enqueue(trainer);
            logger.LogDebug("Pool slot {Index} ready.", i + 1);
        }

        logger.LogInformation("Inference pool ready — {Size} slots available.", _total);
    }

    /// <summary>
    /// Acquires exclusive ownership of a <see cref="MACETrain"/> instance.
    /// Blocks if all slots are busy. Dispose the returned <see cref="PooledInference"/>
    /// to release the slot back to the pool.
    /// </summary>
    public async Task<PooledInference> AcquireAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        await _semaphore.WaitAsync(ct);

        if (_available.TryDequeue(out var trainer))
            return new PooledInference(trainer, this);

        // Invariant violation — should never happen
        _semaphore.Release();
        throw new InvalidOperationException("Pool invariant violated: semaphore granted but queue empty.");
    }

    internal void Return(MACETrain instance)
    {
        _available.Enqueue(instance);
        _semaphore.Release();
    }

    public int Available => _available.Count;
    public int Total     => _total;

    public void Dispose()
    {
        if (!_disposed)
        {
            _semaphore.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Scoped lease on a <see cref="MACETrain"/> instance.
/// Dispose to return the instance to the pool.
/// </summary>
public readonly struct PooledInference : IDisposable
{
    public MACETrain Inferencer { get; }
    private readonly InferencePool _pool;

    internal PooledInference(MACETrain inferencer, InferencePool pool)
    {
        Inferencer = inferencer;
        _pool      = pool;
    }

    public void Dispose() => _pool.Return(Inferencer);
}
