using System.Collections.Concurrent;
using System.Diagnostics;
using Microsoft.Extensions.Options;

namespace MACE.Core;

/// <summary>
/// Thread-safe pool of pre-warmed <see cref="MACETrain"/> instances.
///
/// Infer.NET's InferenceEngine is not thread-safe: Infer&lt;T&gt;(), ObservedValue
/// assignment, and InitialiseTo() all mutate internal state. Each concurrent
/// gRPC call must own its instance exclusively.
///
/// Creating a new MACETrain per request is too expensive because CreateModel()
/// compiles the Infer.NET factor graph (~1–3 s first slot; subsequent slots
/// reuse the compiled code and are much faster). The pool pays this cost once
/// at startup and leases slots to requests.
/// </summary>
public sealed class InferencePool : IInferencePool, IDisposable
{
    private readonly SemaphoreSlim              _semaphore;
    private readonly ConcurrentQueue<MACETrain> _available;
    private readonly ILogger<InferencePool>     _logger;
    private readonly int                        _total;
    private bool _disposed;

    public InferencePool(IOptions<InferenceOptions> options, ILogger<InferencePool> logger)
    {
        _logger    = logger;
        var opts   = options.Value;
        _total     = opts.PoolSize;
        _semaphore = new SemaphoreSlim(_total, _total);
        _available = new ConcurrentQueue<MACETrain>();

        logger.LogInformation(
            "Warming up pool: {Size} slot(s) × ({Sensors} sensor types, {Levels} threat levels)",
            _total, opts.NumSensorTypes, opts.NumThreatLevels);

        var totalSw = Stopwatch.StartNew();

        for (int i = 0; i < _total; i++)
        {
            var slotSw = Stopwatch.StartNew();
            var trainer = new MACETrain(opts.NumSensorTypes, opts.NumThreatLevels);
            trainer.CreateModel();
            slotSw.Stop();
            _available.Enqueue(trainer);

            // The first slot pays the full Infer.NET JIT + Roslyn compilation cost.
            // Subsequent slots reuse the compiled algorithm code and are much faster.
            var note = i == 0 ? "  ← includes Infer.NET JIT / Roslyn compilation" : string.Empty;
            logger.LogInformation("Slot {Slot}/{Total} ready  {Ms}ms{Note}",
                i + 1, _total, slotSw.ElapsedMilliseconds, note);
        }

        totalSw.Stop();
        logger.LogInformation(
            "Pool ready: {Available}/{Total} slots  total warmup {TotalMs}ms",
            _total, _total, totalSw.ElapsedMilliseconds);
    }

    /// <summary>
    /// Acquires exclusive ownership of a <see cref="MACETrain"/> instance.
    /// Blocks until a slot is free or the cancellation token fires.
    /// Dispose the returned <see cref="PooledInference"/> to release the slot.
    /// </summary>
    public async Task<PooledInference> AcquireAsync(CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        await _semaphore.WaitAsync(ct);

        if (_available.TryDequeue(out var trainer))
            return new PooledInference(trainer, () => Return(trainer));

        // Invariant violation — semaphore granted but queue empty.
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
            _logger.LogInformation("Pool disposed.");
        }
    }
}

/// <summary>
/// Scoped lease on a <see cref="MACETrain"/> instance.
/// Returning it to the pool on Dispose() is deterministic via the using pattern.
/// The <paramref name="release"/> action decouples this struct from the concrete
/// pool type, allowing tests to construct leases with a no-op action.
/// </summary>
public readonly struct PooledInference : IDisposable
{
    public MACETrain Inferencer { get; }
    private readonly Action _release;

    internal PooledInference(MACETrain inferencer, Action release)
    {
        Inferencer = inferencer;
        _release   = release;
    }

    public void Dispose() => _release();
}
