using FluentAssertions;
using Grpc.Core;
using MACE;
using MACE.Core;
using MACE.Protos;
using MACE.Services;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Microsoft.ML.Probabilistic.Distributions;
using Moq;
using Xunit;

namespace MACE.Tests.Services;

// ── Fixture ───────────────────────────────────────────────────────────────────
// Owns a pre-compiled MACETrain used only by happy-path tests that need real
// inference.  All other tests use a mock pool so no Infer.NET is involved.

public sealed class GrpcServiceFixture : IDisposable
{
    public const int NumSensorTypes = 3;
    public const int NumThreatLevels = 5;
    public const int MinSensors     = 2;

    public MACETrain Trainer { get; }

    public GrpcServiceFixture()
    {
        Trainer = new MACETrain(NumSensorTypes, NumThreatLevels);
        Trainer.CreateModel();
        // Pre-compile the Infer.NET algorithm.
        Trainer.InferOnline([2, -1, -1], UniformPriors());
    }

    // ── Service factory ───────────────────────────────────────────────────────

    public MaceInferenceGrpcService BuildService(IInferencePool pool, int minSensors = MinSensors)
    {
        var opts = Options.Create(new InferenceOptions
        {
            NumSensorTypes         = NumSensorTypes,
            NumThreatLevels          = NumThreatLevels,
            PoolSize               = 1,
            MinSensorsForInference = minSensors,
            PoolAcquireTimeoutMs   = 500
        });
        return new MaceInferenceGrpcService(
            pool,
            new PriorUpdateService(),
            opts,
            NullLogger<MaceInferenceGrpcService>.Instance);
    }

    // ── Mock pool helpers ─────────────────────────────────────────────────────

    /// Pool that does nothing — used for tests that never reach AcquireAsync.
    public static Mock<IInferencePool> IdlePool()
    {
        var m = new Mock<IInferencePool>();
        m.SetupGet(p => p.Available).Returns(1);
        m.SetupGet(p => p.Total).Returns(1);
        return m;
    }

    /// Pool that returns a working lease backed by the fixture's MACETrain.
    public Mock<IInferencePool> WorkingPool()
    {
        // PooledInference(trainer, release) — release is a no-op in tests.
        var lease = new PooledInference(Trainer, () => { });
        var m = new Mock<IInferencePool>();
        m.SetupGet(p => p.Available).Returns(1);
        m.SetupGet(p => p.Total).Returns(1);
        m.Setup(p => p.AcquireAsync(It.IsAny<CancellationToken>()))
         .ReturnsAsync(lease);
        return m;
    }

    /// Pool that throws OperationCanceledException, simulating acquire timeout.
    public static Mock<IInferencePool> ExhaustedPool()
    {
        var m = new Mock<IInferencePool>();
        m.SetupGet(p => p.Available).Returns(0);
        m.SetupGet(p => p.Total).Returns(1);
        m.Setup(p => p.AcquireAsync(It.IsAny<CancellationToken>()))
         .ThrowsAsync(new OperationCanceledException());
        return m;
    }

    // ── Context & request helpers ─────────────────────────────────────────────

    /// Minimal ServerCallContext — the service only reads CancellationToken,
    /// which Moq returns as CancellationToken.None (never cancelled) by default.
    public static ServerCallContext Ctx() =>
        new Mock<ServerCallContext>().Object;

    /// A fully valid InferRequest with the given sensor readings.
    public static InferRequest ValidInferRequest(int[] sensorReadings, double[]? warmStart = null)
    {
        var req = new InferRequest { IncidentId = "test-inc" };
        req.SensorReadings.AddRange(sensorReadings);

        for (int i = 0; i < NumSensorTypes; i++)
        {
            req.ThetaPriors.Add(new BetaParams { Alpha = 1.0, Beta = 9.0 });
            var phi = new DirichletParams();
            phi.Pseudocounts.AddRange(Enumerable.Repeat(1.0, NumThreatLevels));
            req.PhiPriors.Add(phi);
        }

        if (warmStart is not null)
            req.WarmStart.AddRange(warmStart);

        return req;
    }

    public static ModelData UniformPriors() => new()
    {
        ThetaDist = Enumerable.Repeat(new Beta(1.0, 1.0), NumSensorTypes).ToArray(),
        PhiDist   = Enumerable.Repeat(
            new Dirichlet([1.0, 1.0, 1.0, 1.0, 1.0]),
            NumSensorTypes).ToArray()
    };

    public void Dispose() { }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

[Trait("Category", "Integration")]
public class MaceInferenceGrpcServiceTests : IClassFixture<GrpcServiceFixture>
{
    private readonly GrpcServiceFixture _fx;

    public MaceInferenceGrpcServiceTests(GrpcServiceFixture fixture) => _fx = fixture;

    // ── Infer: request validation ─────────────────────────────────────────────
    // Validation runs before pool acquisition, so the idle pool is enough.

    [Fact]
    public async Task Infer_WrongAnnotationCount_ThrowsInvalidArgument()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3]); // 2 instead of 3

        Func<Task> act = () => svc.Infer(req, GrpcServiceFixture.Ctx());
        (await act.Should().ThrowAsync<RpcException>())
            .Which.StatusCode.Should().Be(StatusCode.InvalidArgument);
    }

    [Fact]
    public async Task Infer_WrongThetaPriorCount_ThrowsInvalidArgument()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, 3]);
        req.ThetaPriors.RemoveAt(0); // now 2 instead of 3

        Func<Task> act = () => svc.Infer(req, GrpcServiceFixture.Ctx());
        (await act.Should().ThrowAsync<RpcException>())
            .Which.StatusCode.Should().Be(StatusCode.InvalidArgument);
    }

    [Fact]
    public async Task Infer_WrongPhiPriorCount_ThrowsInvalidArgument()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, 3]);
        req.PhiPriors.RemoveAt(0);

        Func<Task> act = () => svc.Infer(req, GrpcServiceFixture.Ctx());
        (await act.Should().ThrowAsync<RpcException>())
            .Which.StatusCode.Should().Be(StatusCode.InvalidArgument);
    }

    [Theory]
    [InlineData(0.0,  9.0)]   // alpha == 0
    [InlineData(-1.0, 9.0)]   // alpha < 0
    [InlineData(1.0,  0.0)]   // beta == 0
    [InlineData(1.0, -1.0)]   // beta < 0
    public async Task Infer_NonPositiveBetaParams_ThrowsInvalidArgument(
        double alpha, double beta)
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, 3]);
        req.ThetaPriors[0] = new BetaParams { Alpha = alpha, Beta = beta };

        Func<Task> act = () => svc.Infer(req, GrpcServiceFixture.Ctx());
        (await act.Should().ThrowAsync<RpcException>())
            .Which.StatusCode.Should().Be(StatusCode.InvalidArgument);
    }

    [Fact]
    public async Task Infer_WrongPhiPseudocountLength_ThrowsInvalidArgument()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, 3]);
        // Replace first phi prior with one that has wrong pseudocount length
        req.PhiPriors[0] = new DirichletParams();
        req.PhiPriors[0].Pseudocounts.AddRange([1.0, 1.0]); // 2 instead of 5

        Func<Task> act = () => svc.Infer(req, GrpcServiceFixture.Ctx());
        (await act.Should().ThrowAsync<RpcException>())
            .Which.StatusCode.Should().Be(StatusCode.InvalidArgument);
    }

    [Fact]
    public async Task Infer_NonPositivePhiPseudocount_ThrowsInvalidArgument()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, 3]);
        req.PhiPriors[0] = new DirichletParams();
        req.PhiPriors[0].Pseudocounts.AddRange([1.0, 0.0, 1.0, 1.0, 1.0]); // zero entry

        Func<Task> act = () => svc.Infer(req, GrpcServiceFixture.Ctx());
        (await act.Should().ThrowAsync<RpcException>())
            .Which.StatusCode.Should().Be(StatusCode.InvalidArgument);
    }

    [Fact]
    public async Task Infer_WarmStartWrongLength_ThrowsInvalidArgument()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        // warm_start is non-empty but has 3 entries instead of 5
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, 3], warmStart: [0.2, 0.2, 0.6]);

        Func<Task> act = () => svc.Infer(req, GrpcServiceFixture.Ctx());
        (await act.Should().ThrowAsync<RpcException>())
            .Which.StatusCode.Should().Be(StatusCode.InvalidArgument);
    }

    // ── Infer: fallback path (< MinSensorsForInference) ──────────────────────
    // The service returns before touching the pool, so the idle mock is fine.

    [Fact]
    public async Task Infer_BelowMinSensors_ReturnsUniformThreatDist()
    {
        // Only 1 annotation present when min is 2 → fallback
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, -1, -1]); // 1 of 3

        var resp = await svc.Infer(req, GrpcServiceFixture.Ctx());

        resp.ThreatDist.Should().HaveCount(GrpcServiceFixture.NumThreatLevels);
        // Fallback distributes probability uniformly
        var expected = 1.0 / GrpcServiceFixture.NumThreatLevels;
        resp.ThreatDist.Should().AllSatisfy(p => p.Should().BeApproximately(expected, 1e-10));
    }

    [Fact]
    public async Task Infer_BelowMinSensors_InferenceMsIsZero()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, -1, -1]);

        var resp = await svc.Infer(req, GrpcServiceFixture.Ctx());

        resp.InferenceMs.Should().Be(0);
    }

    [Fact]
    public async Task Infer_BelowMinSensors_ThreatLevelIsMaxSensorReading()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        // Only sensor 0 fires with label 3
        var req = GrpcServiceFixture.ValidInferRequest([3, -1, -1]);

        var resp = await svc.Infer(req, GrpcServiceFixture.Ctx());

        resp.ThreatLevel.Should().Be(3);
    }

    [Fact]
    public async Task Infer_BelowMinSensors_SensorReliabilityIsEmpty()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, -1, -1]);

        var resp = await svc.Infer(req, GrpcServiceFixture.Ctx());

        resp.SensorReliability.Should().BeEmpty();
    }

    // ── Infer: pool exhaustion ────────────────────────────────────────────────

    [Fact]
    public async Task Infer_PoolExhausted_ThrowsUnavailable()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.ExhaustedPool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, -1]); // 2 present → passes validation

        Func<Task> act = () => svc.Infer(req, GrpcServiceFixture.Ctx());
        (await act.Should().ThrowAsync<RpcException>())
            .Which.StatusCode.Should().Be(StatusCode.Unavailable);
    }

    // ── Infer: happy path (real inference) ────────────────────────────────────

    [Fact]
    public async Task Infer_ValidRequest_ThreatDistSumsToOne()
    {
        var svc = _fx.BuildService(_fx.WorkingPool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, 3]);

        var resp = await svc.Infer(req, GrpcServiceFixture.Ctx());

        resp.ThreatDist.Sum().Should().BeApproximately(1.0, precision: 1e-6);
    }

    [Fact]
    public async Task Infer_ValidRequest_SensorReliabilityOnlyContainsPresentSensors()
    {
        var svc = _fx.BuildService(_fx.WorkingPool().Object);
        // Sensor 2 is absent — it must not appear in sensor_reliability
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, -1]);

        var resp = await svc.Infer(req, GrpcServiceFixture.Ctx());

        resp.SensorReliability.Should().HaveCount(2); // sensors 0 and 1
        resp.SensorReliability.Should().NotContain(sr => sr.SensorReading == -1);
    }

    [Fact]
    public async Task Infer_ValidRequest_IncidentIdIsEchoedBack()
    {
        var svc = _fx.BuildService(_fx.WorkingPool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, 3]);
        req.IncidentId = "lobby-intrusion-042";

        var resp = await svc.Infer(req, GrpcServiceFixture.Ctx());

        resp.IncidentId.Should().Be("lobby-intrusion-042");
    }

    [Fact]
    public async Task Infer_ValidRequest_NumObservationsMatchesPresentAnnotations()
    {
        var svc = _fx.BuildService(_fx.WorkingPool().Object);
        var req = GrpcServiceFixture.ValidInferRequest([3, 3, -1]); // 2 present

        var resp = await svc.Infer(req, GrpcServiceFixture.Ctx());

        resp.NumObservations.Should().Be(2);
    }

    // ── UpdatePriors ──────────────────────────────────────────────────────────

    [Fact]
    public async Task UpdatePriors_UnknownVerdict_ThrowsInvalidArgument()
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = new UpdatePriorsRequest { Verdict = "MAYBE_ALARM", LearningRate = 0.5 };
        req.Sensors.Add(new SensorPriorUpdate
        {
            SensorTypeIndex = 0, SensorReading = 3, FaultProbMean = 0.1,
            CurrentTheta    = new BetaParams { Alpha = 1.0, Beta = 9.0 }
        });

        Func<Task> act = () => svc.UpdatePriors(req, GrpcServiceFixture.Ctx());
        (await act.Should().ThrowAsync<RpcException>())
            .Which.StatusCode.Should().Be(StatusCode.InvalidArgument);
    }

    [Theory]
    [InlineData("TRUE_ALARM")]
    [InlineData("FALSE_ALARM")]
    public async Task UpdatePriors_ValidVerdict_UpdatedThetasCountMatchesSensors(string verdict)
    {
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = new UpdatePriorsRequest { Verdict = verdict, LearningRate = 0.5 };
        req.Sensors.Add(new SensorPriorUpdate
        {
            SensorTypeIndex = 0, SensorReading = 3, FaultProbMean = 0.1,
            CurrentTheta    = new BetaParams { Alpha = 1.0, Beta = 9.0 }
        });
        req.Sensors.Add(new SensorPriorUpdate
        {
            SensorTypeIndex = 1, SensorReading = 3, FaultProbMean = 0.1,
            CurrentTheta    = new BetaParams { Alpha = 1.0, Beta = 9.0 }
        });

        var resp = await svc.UpdatePriors(req, GrpcServiceFixture.Ctx());

        resp.UpdatedThetas.Should().HaveCount(2);
    }

    [Fact]
    public async Task UpdatePriors_DefaultLearningRate_UsesHalfPointFive()
    {
        // LearningRate = 0 in the request → service falls back to 0.5
        var svc = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var req = new UpdatePriorsRequest { Verdict = "TRUE_ALARM", LearningRate = 0 };
        req.Sensors.Add(new SensorPriorUpdate
        {
            SensorTypeIndex = 0, SensorReading = 3, FaultProbMean = 0.1,
            CurrentTheta    = new BetaParams { Alpha = 1.0, Beta = 9.0 }
        });

        var resp = await svc.UpdatePriors(req, GrpcServiceFixture.Ctx());

        // With lr=0.5 and TrueAlarm+flagged: beta += 0.5 * (1 - 0.1) = 0.45
        resp.UpdatedThetas[0].Beta.Should().BeApproximately(9.45, precision: 1e-10);
    }

    // ── Health ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Health_ReturnsPoolAvailableAndTotal()
    {
        var mockPool = GrpcServiceFixture.IdlePool();
        mockPool.SetupGet(p => p.Available).Returns(3);
        mockPool.SetupGet(p => p.Total).Returns(4);

        var svc  = _fx.BuildService(mockPool.Object);
        var resp = await svc.Health(new HealthRequest(), GrpcServiceFixture.Ctx());

        resp.PoolAvailable.Should().Be(3);
        resp.PoolTotal.Should().Be(4);
    }

    [Fact]
    public async Task Health_StatusIsHealthy()
    {
        var svc  = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var resp = await svc.Health(new HealthRequest(), GrpcServiceFixture.Ctx());

        resp.Status.Should().Be("healthy");
    }

    [Fact]
    public async Task Health_UptimeSecondsIsNonNegative()
    {
        var svc  = _fx.BuildService(GrpcServiceFixture.IdlePool().Object);
        var resp = await svc.Health(new HealthRequest(), GrpcServiceFixture.Ctx());

        resp.UptimeSeconds.Should().BeGreaterThanOrEqualTo(0);
    }
}
