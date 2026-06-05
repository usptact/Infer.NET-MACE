using FluentAssertions;
using MACE;
using MACE.Core;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Microsoft.ML.Probabilistic.Distributions;
using Xunit;

namespace MACE.Tests.Core;

// ── Shared fixture ────────────────────────────────────────────────────────────
// Creates a 2-slot InferencePool that is shared across every test in the class.
// Pool creation calls MACETrain.CreateModel() for each slot; Infer.NET caches
// the compiled algorithm globally, so subsequent pools with the same dimensions
// (e.g. those created for isolation tests) warm up much faster.
//
// Every test that acquires leases must release them in a finally block so the
// pool is fully available for the next test (xUnit runs within-class tests
// sequentially, but assertion failures must not leak acquired slots).

public sealed class InferencePoolFixture : IDisposable
{
    public const int PoolSize  = 2;
    public const int SensorTypes = 3;
    public const int ThreatLevels = 5;

    public InferencePool Pool { get; }

    public InferencePoolFixture() => Pool = BuildPool(PoolSize);

    /// <summary>Creates an independent pool with the given slot count.</summary>
    public static InferencePool BuildPool(int size) =>
        new(Options.Create(new InferenceOptions
        {
            NumSensorTypes         = SensorTypes,
            NumThreatLevels        = ThreatLevels,
            PoolSize               = size,
            MinSensorsForInference = 2,
            PoolAcquireTimeoutMs   = 500
        }),
        NullLogger<InferencePool>.Instance);

    public static ModelData UniformPriors() => new()
    {
        ThetaDist = Enumerable.Repeat(new Beta(1.0, 1.0), SensorTypes).ToArray(),
        PhiDist   = Enumerable.Repeat(
            new Dirichlet([1.0, 1.0, 1.0, 1.0, 1.0]),
            SensorTypes).ToArray()
    };

    public void Dispose() => Pool.Dispose();
}

// ── Test class ────────────────────────────────────────────────────────────────

[Trait("Category", "Integration")]
public class InferencePoolTests : IClassFixture<InferencePoolFixture>
{
    private readonly InferencePoolFixture _fx;
    private InferencePool Pool => _fx.Pool;

    public InferencePoolTests(InferencePoolFixture fixture) => _fx = fixture;

    // ── Initial state ─────────────────────────────────────────────────────────

    [Fact]
    public void Total_ReflectsConfiguredPoolSize()
    {
        Pool.Total.Should().Be(InferencePoolFixture.PoolSize);
    }

    [Fact]
    public void Available_InitiallyEqualsTotal()
    {
        Pool.Available.Should().Be(Pool.Total);
    }

    // ── Single acquire / release cycle ────────────────────────────────────────

    [Fact]
    public async Task AcquireAsync_DecrementsAvailable()
    {
        var lease = await Pool.AcquireAsync();
        try
        {
            Pool.Available.Should().Be(InferencePoolFixture.PoolSize - 1);
        }
        finally
        {
            lease.Dispose();
        }
    }

    [Fact]
    public async Task PooledInference_Dispose_IncrementsAvailableBack()
    {
        int before = Pool.Available;
        var lease  = await Pool.AcquireAsync();
        lease.Dispose();

        Pool.Available.Should().Be(before);
    }

    [Fact]
    public async Task AcquireAsync_LeasedInferencer_IsNotNull()
    {
        var lease = await Pool.AcquireAsync();
        try
        {
            lease.Inferencer.Should().NotBeNull();
        }
        finally
        {
            lease.Dispose();
        }
    }

    [Fact]
    public async Task AcquireAsync_LeasedInferencer_IsReadyToRunInference()
    {
        // Verifies that every pooled MACETrain had CreateModel() called on it
        // and can produce a valid probability distribution.
        using var lease = await Pool.AcquireAsync();
        var result = lease.Inferencer.InferOnline(
            [3, 3, 3], InferencePoolFixture.UniformPriors());

        result.ThreatDist.GetProbs().ToArray().Sum()
              .Should().BeApproximately(1.0, precision: 1e-6);
    }

    // ── Full-pool acquire / release ───────────────────────────────────────────

    [Fact]
    public async Task AcquireAsync_AllSlots_CanBeAcquiredSequentially()
    {
        var leases = new PooledInference[InferencePoolFixture.PoolSize];
        try
        {
            for (int i = 0; i < leases.Length; i++)
                leases[i] = await Pool.AcquireAsync();

            Pool.Available.Should().Be(0);
        }
        finally
        {
            foreach (var l in leases) l.Dispose();
        }
    }

    [Fact]
    public async Task AcquireAsync_AfterFullOccupancyRelease_PoolIsFullAgain()
    {
        var leases = new PooledInference[InferencePoolFixture.PoolSize];
        for (int i = 0; i < leases.Length; i++)
            leases[i] = await Pool.AcquireAsync();

        foreach (var l in leases) l.Dispose();

        Pool.Available.Should().Be(Pool.Total);
    }

    // ── Pool exhaustion ───────────────────────────────────────────────────────

    [Fact]
    public async Task AcquireAsync_BeyondCapacity_CancelsWhenTokenFires()
    {
        var leases = new PooledInference[InferencePoolFixture.PoolSize];
        for (int i = 0; i < leases.Length; i++)
            leases[i] = await Pool.AcquireAsync();

        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
            Func<Task> act = () => Pool.AcquireAsync(cts.Token);
            await act.Should().ThrowAsync<OperationCanceledException>();
        }
        finally
        {
            foreach (var l in leases) l.Dispose();
        }
    }

    // ── Concurrent access ─────────────────────────────────────────────────────

    [Fact]
    public async Task AcquireAsync_ConcurrentTasks_AllSucceedUpToCapacity()
    {
        // Fire N tasks simultaneously (N == pool size). Task.WhenAll must
        // resolve all of them — not timeout or deadlock.
        var tasks = Enumerable
            .Range(0, InferencePoolFixture.PoolSize)
            .Select(_ => Pool.AcquireAsync())
            .ToArray();

        var leases = await Task.WhenAll(tasks);
        try
        {
            leases.Should().HaveCount(InferencePoolFixture.PoolSize);
            Pool.Available.Should().Be(0);
        }
        finally
        {
            foreach (var l in leases) l.Dispose();
        }

        Pool.Available.Should().Be(Pool.Total);
    }

    [Fact]
    public async Task AcquireAsync_ExtraTaskUnblocks_WhenSlotIsReleased()
    {
        // Fill pool to capacity.
        var leases = new PooledInference[InferencePoolFixture.PoolSize];
        for (int i = 0; i < leases.Length; i++)
            leases[i] = await Pool.AcquireAsync();

        // Start an extra acquire — SemaphoreSlim.WaitAsync returns an
        // incomplete Task immediately when count is 0.
        var waiting = Pool.AcquireAsync();
        waiting.IsCompleted.Should().BeFalse("pool is fully occupied");

        // Release one slot; the waiting task must unblock within 2 s.
        leases[0].Dispose();
        var extra = await waiting.WaitAsync(TimeSpan.FromSeconds(2));
        extra.Dispose();

        // Release the remaining leases.
        for (int i = 1; i < leases.Length; i++) leases[i].Dispose();

        Pool.Available.Should().Be(Pool.Total);
    }

    // ── Dispose semantics ─────────────────────────────────────────────────────

    [Fact]
    public async Task AcquireAsync_OnDisposedPool_ThrowsObjectDisposedException()
    {
        // Isolated pool — disposing it must not affect the shared fixture pool.
        var isolated = InferencePoolFixture.BuildPool(size: 1);
        isolated.Dispose();

        Func<Task> act = () => isolated.AcquireAsync();
        await act.Should().ThrowAsync<ObjectDisposedException>();
    }

    [Fact]
    public void Dispose_CalledTwice_DoesNotThrow()
    {
        // InferencePool.Dispose() guards with `if (!_disposed)` — second call is a no-op.
        var isolated = InferencePoolFixture.BuildPool(size: 1);
        isolated.Dispose();

        Action act = isolated.Dispose;
        act.Should().NotThrow();
    }
}
