using FluentAssertions;
using MACE;
using Microsoft.ML.Probabilistic.Distributions;
using Xunit;

namespace MACE.Tests.Inference;

// ── Shared fixture ────────────────────────────────────────────────────────────
// Creates one MACETrain(3 sensors, 5 categories) instance for the whole class.
// The warm-up call in the constructor triggers Infer.NET's Roslyn JIT so that
// individual tests don't pay the first-call compilation cost (~1-3 s).
// xUnit runs tests within a class sequentially, so sharing the mutable trainer
// is safe — every InferOnline/InferModelData call fully reinitialises state.

public sealed class MACETrainFixture : IDisposable
{
    public const int NumSensorTypes = 3;
    public const int NumThreatLevels = 5;

    public MACETrain Trainer { get; }

    public MACETrainFixture()
    {
        Trainer = new MACETrain(NumSensorTypes, NumThreatLevels);
        Trainer.CreateModel();
        // Pre-compile so tests run at steady-state speed.
        Trainer.InferOnline([2, -1, -1], UniformPriors());
    }

    // Beta(1,1): flat prior — no knowledge of sensor reliability
    public static ModelData UniformPriors() => new()
    {
        ThetaDist = Enumerable.Repeat(new Beta(1.0, 1.0), NumSensorTypes).ToArray(),
        PhiDist   = Enumerable.Repeat(
            new Dirichlet([1.0, 1.0, 1.0, 1.0, 1.0]),
            NumSensorTypes).ToArray()
    };

    // Beta(1,9): strong prior that sensors are reliable (90 % chance of not spamming)
    public static ModelData ReliablePriors() => new()
    {
        ThetaDist = Enumerable.Repeat(new Beta(1.0, 9.0), NumSensorTypes).ToArray(),
        PhiDist   = Enumerable.Repeat(
            new Dirichlet([1.0, 1.0, 1.0, 1.0, 1.0]),
            NumSensorTypes).ToArray()
    };

    public void Dispose() { }
}

// ── Test class ────────────────────────────────────────────────────────────────

[Trait("Category", "Integration")]
public class MACETrainTests : IClassFixture<MACETrainFixture>
{
    private readonly MACETrainFixture _fx;
    private MACETrain Trainer => _fx.Trainer;

    public MACETrainTests(MACETrainFixture fixture) => _fx = fixture;

    // ── Constructor validation ────────────────────────────────────────────────
    // These instantiate fresh MACETrain objects but never call CreateModel(),
    // so they are cheap even though they carry the Integration trait.

    [Theory]
    [InlineData(0,  3, 5)]
    [InlineData(-1, 3, 5)]
    public void Constructor_NonPositiveWorkers_ThrowsArgumentOutOfRange(
        int workers, int items, int cats)
    {
        var act = () => new MACETrain(workers, items, cats);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Theory]
    [InlineData(3, 0,  5)]
    [InlineData(3, -1, 5)]
    public void Constructor_NonPositiveNumItems_ThrowsArgumentOutOfRange(
        int workers, int items, int cats)
    {
        var act = () => new MACETrain(workers, items, cats);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Theory]
    [InlineData(3, 3, 0)]
    [InlineData(3, 3, -1)]
    public void Constructor_NonPositiveCategories_ThrowsArgumentOutOfRange(
        int workers, int items, int cats)
    {
        var act = () => new MACETrain(workers, items, cats);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_Online_NonPositiveSensorTypes_ThrowsArgumentOutOfRange(
        int sensorTypes)
    {
        var act = () => new MACETrain(sensorTypes, numThreatLevels: 5);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ── SetModelData ──────────────────────────────────────────────────────────

    [Fact]
    public void SetModelData_Null_ThrowsArgumentNullException()
    {
        var act = () => Trainer.SetModelData(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    // ── InitializeLabels ──────────────────────────────────────────────────────

    [Theory]
    [InlineData(0,  5)]
    [InlineData(-1, 5)]
    public void InitializeLabels_NonPositiveNumIncidents_ThrowsArgumentOutOfRange(
        int numIncidents, int numThreatLevels)
    {
        var act = () => Trainer.InitializeLabels(numIncidents, numThreatLevels);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Theory]
    [InlineData(1, 0)]
    [InlineData(1, -1)]
    public void InitializeLabels_NonPositiveNumThreatLevels_ThrowsArgumentOutOfRange(
        int numIncidents, int numThreatLevels)
    {
        var act = () => Trainer.InitializeLabels(numIncidents, numThreatLevels);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void InitializeLabels_WrongLengthWarmStart_FallsBackToRandomWithoutThrowing()
    {
        // warmStart.Length (2) != numItems (1) → silent fallback to random init
        var wrongLength = new[]
        {
            Discrete.PointMass(3, MACETrainFixture.NumThreatLevels),
            Discrete.PointMass(3, MACETrainFixture.NumThreatLevels)
        };

        var act = () => Trainer.InitializeLabels(
            numIncidents:  1,
            numThreatLevels: MACETrainFixture.NumThreatLevels,
            warmStart:     wrongLength);

        act.Should().NotThrow();
    }

    // ── InferOnline: input validation ─────────────────────────────────────────

    [Fact]
    public void InferOnline_NullAnnotations_ThrowsArgumentNullException()
    {
        var act = () => Trainer.InferOnline(null!, MACETrainFixture.UniformPriors());
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void InferOnline_NullPriors_ThrowsArgumentNullException()
    {
        var act = () => Trainer.InferOnline([3, 3, 3], null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void InferOnline_AnnotationsLengthMismatch_ThrowsArgumentException()
    {
        // Model expects 3 annotations; 4 provided
        var act = () => Trainer.InferOnline([3, 3, 3, 3], MACETrainFixture.UniformPriors());
        act.Should().Throw<ArgumentException>();
    }

    // ── InferOnline: output structure ─────────────────────────────────────────

    [Fact]
    public void InferOnline_ThreatDistProbabilitiesSumToOne()
    {
        var result = Trainer.InferOnline([3, 3, 3], MACETrainFixture.UniformPriors());
        var probs  = result.ThreatDist.GetProbs().ToArray();

        probs.Sum().Should().BeApproximately(1.0, precision: 1e-6);
    }

    [Fact]
    public void InferOnline_ConfidenceEqualsMaxThreatDistProbability()
    {
        var result = Trainer.InferOnline([3, 3, 3], MACETrainFixture.UniformPriors());
        var max    = result.ThreatDist.GetProbs().ToArray().Max();

        result.Confidence.Should().BeApproximately(max, precision: 1e-10);
    }

    [Fact]
    public void InferOnline_ThreatLevelIsArgmaxOfThreatDist()
    {
        var result = Trainer.InferOnline([3, 3, 3], MACETrainFixture.UniformPriors());
        var probs  = result.ThreatDist.GetProbs().ToArray();
        var argmax = Array.IndexOf(probs, probs.Max());

        result.ThreatLevel.Should().Be(argmax);
    }

    [Fact]
    public void InferOnline_EntropyIsNonNegative()
    {
        var result = Trainer.InferOnline([2, 2, 2], MACETrainFixture.UniformPriors());
        result.Entropy.Should().BeGreaterThanOrEqualTo(0.0);
    }

    [Fact]
    public void InferOnline_FaultDistLengthMatchesSensorCount()
    {
        var result = Trainer.InferOnline([3, -1, 3], MACETrainFixture.UniformPriors());
        result.FaultDist.Should().HaveCount(MACETrainFixture.NumSensorTypes);
    }

    // ── InferOnline: inference quality ───────────────────────────────────────
    // These use reliable priors (Beta(1,9)) to get a strong enough signal that
    // VMP converges correctly regardless of the random label initialisation.

    [Fact]
    public void InferOnline_AllSensorsAgree_ThreatLevelMatchesAnnotation()
    {
        var result = Trainer.InferOnline([3, 3, 3], MACETrainFixture.ReliablePriors());
        result.ThreatLevel.Should().Be(3);
    }

    [Fact]
    public void InferOnline_AllSensorsAgree_ConfidenceIsHigh()
    {
        var result = Trainer.InferOnline([3, 3, 3], MACETrainFixture.ReliablePriors());
        result.Confidence.Should().BeGreaterThan(0.9);
    }

    [Fact]
    public void InferOnline_ConsensusEntropyIsLowerThanDisagreementEntropy()
    {
        // Consensus: all 3 sensors at MEDIUM(2) → model is certain → low entropy
        // Disagreement: sensors at CLEAR(0), CRITICAL(4), MEDIUM(2) → model is uncertain → high entropy
        var consensus    = Trainer.InferOnline([2, 2, 2], MACETrainFixture.UniformPriors());
        var disagreement = Trainer.InferOnline([0, 4, 2], MACETrainFixture.UniformPriors());

        consensus.Entropy.Should().BeLessThan(disagreement.Entropy);
    }

    [Fact]
    public void InferOnline_SinglePresentAnnotation_StillProducesValidDistribution()
    {
        // Only sensor 0 reports MEDIUM(2); others absent
        var result = Trainer.InferOnline([2, -1, -1], MACETrainFixture.UniformPriors());
        var probs  = result.ThreatDist.GetProbs().ToArray();

        probs.Sum().Should().BeApproximately(1.0, precision: 1e-6);
    }

    // ── InferOnline: warm-start ───────────────────────────────────────────────

    [Fact]
    public void InferOnline_WithWarmStart_ProducesValidDistribution()
    {
        var warmStart = Discrete.PointMass(3, MACETrainFixture.NumThreatLevels);
        var result    = Trainer.InferOnline([3, 3, 3], MACETrainFixture.ReliablePriors(), warmStart);
        var probs     = result.ThreatDist.GetProbs().ToArray();

        probs.Sum().Should().BeApproximately(1.0, precision: 1e-6);
    }

    [Fact]
    public void InferOnline_WarmStartBiasedToCorrectLabel_SameThreatLevelAsNoWarmStart()
    {
        // For a strong 3-sensor consensus, warm-start direction shouldn't change the outcome.
        var warmStart = Discrete.PointMass(3, MACETrainFixture.NumThreatLevels);
        var withWarm    = Trainer.InferOnline([3, 3, 3], MACETrainFixture.ReliablePriors(), warmStart);
        var withoutWarm = Trainer.InferOnline([3, 3, 3], MACETrainFixture.ReliablePriors());

        withWarm.ThreatLevel.Should().Be(withoutWarm.ThreatLevel);
    }

    // ── InferModelData ────────────────────────────────────────────────────────
    // The fixture's trainer has numItems=1, so a 1-row data matrix is valid.

    [Fact]
    public void InferModelData_Null_ThrowsArgumentNullException()
    {
        Trainer.SetModelData(MACETrainFixture.UniformPriors());
        Trainer.InitializeLabels(1, MACETrainFixture.NumThreatLevels);

        var act = () => Trainer.InferModelData(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void InferModelData_WrongItemCount_ThrowsArgumentException()
    {
        Trainer.SetModelData(MACETrainFixture.UniformPriors());
        Trainer.InitializeLabels(1, MACETrainFixture.NumThreatLevels);

        // Model has numItems=1; 2-row matrix is invalid
        var act = () => Trainer.InferModelData([[3, 3, 3], [1, 1, 1]]);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void InferModelData_WrongWorkerCount_ThrowsArgumentException()
    {
        Trainer.SetModelData(MACETrainFixture.UniformPriors());
        Trainer.InitializeLabels(1, MACETrainFixture.NumThreatLevels);

        // Model expects 3 workers; 4 annotations per item is invalid
        var act = () => Trainer.InferModelData([[3, 3, 3, 3]]);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void InferModelData_ValidData_ReturnsCorrectlyShapedArrays()
    {
        Trainer.SetModelData(MACETrainFixture.UniformPriors());
        Trainer.InitializeLabels(1, MACETrainFixture.NumThreatLevels);

        var result = Trainer.InferModelData([[3, 3, 3]]);

        result.ThreatDist.Should().HaveCount(1);
        result.ThetaDist.Should().HaveCount(MACETrainFixture.NumSensorTypes);
        result.PhiDist.Should().HaveCount(MACETrainFixture.NumSensorTypes);
        result.FaultDist.Should().HaveCount(1);
        result.FaultDist[0].Should().HaveCount(MACETrainFixture.NumSensorTypes);
    }
}
