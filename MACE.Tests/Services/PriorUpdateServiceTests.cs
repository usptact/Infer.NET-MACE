using FluentAssertions;
using MACE.Services;
using Xunit;

namespace MACE.Tests.Services;

public class PriorUpdateServiceTests
{
    private readonly PriorUpdateService _sut = new();

    // ── ParseVerdict ──────────────────────────────────────────────────────────

    [Theory]
    [InlineData("TRUE_ALARM",  Verdict.TrueAlarm)]
    [InlineData("FALSE_ALARM", Verdict.FalseAlarm)]
    public void ParseVerdict_CanonicalInput_ReturnsCorrectVerdict(string input, Verdict expected)
    {
        PriorUpdateService.ParseVerdict(input).Should().Be(expected);
    }

    [Theory]
    [InlineData("true_alarm")]
    [InlineData("True_Alarm")]
    [InlineData("TRUE_ALARM")]
    public void ParseVerdict_CaseVariants_ParseAsTrueAlarm(string input)
    {
        PriorUpdateService.ParseVerdict(input).Should().Be(Verdict.TrueAlarm);
    }

    [Theory]
    [InlineData("false_alarm")]
    [InlineData("False_Alarm")]
    [InlineData("FALSE_ALARM")]
    public void ParseVerdict_CaseVariants_ParseAsFalseAlarm(string input)
    {
        PriorUpdateService.ParseVerdict(input).Should().Be(Verdict.FalseAlarm);
    }

    [Theory]
    [InlineData("")]
    [InlineData("UNKNOWN")]
    [InlineData("alarm")]
    [InlineData("TRUE")]
    public void ParseVerdict_UnknownInput_ThrowsArgumentException(string input)
    {
        var act = () => PriorUpdateService.ParseVerdict(input);
        act.Should().Throw<ArgumentException>();
    }

    // ── UpdateTheta: absent sensor (annotation == -1) ─────────────────────────

    [Theory]
    [InlineData(Verdict.TrueAlarm)]
    [InlineData(Verdict.FalseAlarm)]
    public void UpdateTheta_AbsentSensor_ReturnsUnchangedPrior(Verdict verdict)
    {
        var prior = new BetaParameters(2.0, 8.0);

        var result = _sut.UpdateTheta(prior, spammerProbMean: 0.5, annotation: -1, verdict);

        result.Should().Be(prior);
    }

    // ── UpdateTheta: TRUE_ALARM + flagged threat (annotation >= 2) ────────────

    [Fact]
    public void UpdateTheta_TrueAlarm_FlaggedThreat_IncreasesBeta()
    {
        var prior = new BetaParameters(1.0, 9.0);
        const double sp = 0.1, lr = 0.5;

        var result = _sut.UpdateTheta(prior, sp, annotation: 3, Verdict.TrueAlarm, lr);

        result.Beta.Should().BeApproximately(prior.Beta + lr * (1.0 - sp), precision: 1e-10);
    }

    [Fact]
    public void UpdateTheta_TrueAlarm_FlaggedThreat_AlphaUnchanged()
    {
        var prior = new BetaParameters(1.0, 9.0);

        var result = _sut.UpdateTheta(prior, spammerProbMean: 0.1, annotation: 3, Verdict.TrueAlarm);

        result.Alpha.Should().Be(prior.Alpha);
    }

    [Fact]
    public void UpdateTheta_TrueAlarm_AnnotationExactlyAtThreshold_TreatedAsFlagged()
    {
        // annotation == MediumThreshold (2) must take the flagged-threat path
        var prior = new BetaParameters(1.0, 9.0);

        var result = _sut.UpdateTheta(prior, spammerProbMean: 0.5, annotation: 2, Verdict.TrueAlarm);

        result.Beta.Should().BeGreaterThan(prior.Beta);
        result.Alpha.Should().Be(prior.Alpha);
    }

    // ── UpdateTheta: TRUE_ALARM + NOT flagged (annotation < 2) ───────────────

    [Fact]
    public void UpdateTheta_TrueAlarm_NotFlagged_IncreasesAlpha()
    {
        var prior = new BetaParameters(1.0, 9.0);
        const double lr = 0.5;

        var result = _sut.UpdateTheta(prior, spammerProbMean: 0.1, annotation: 1, Verdict.TrueAlarm, lr);

        result.Alpha.Should().BeApproximately(prior.Alpha + lr * 0.3, precision: 1e-10);
    }

    [Fact]
    public void UpdateTheta_TrueAlarm_NotFlagged_BetaUnchanged()
    {
        var prior = new BetaParameters(1.0, 9.0);

        var result = _sut.UpdateTheta(prior, spammerProbMean: 0.1, annotation: 1, Verdict.TrueAlarm);

        result.Beta.Should().Be(prior.Beta);
    }

    [Fact]
    public void UpdateTheta_TrueAlarm_AnnotationJustBelowThreshold_TakesNotFlaggedPath()
    {
        // annotation == 1 is below MediumThreshold (2); delta must be lr*0.3 on alpha
        var prior = new BetaParameters(1.0, 9.0);

        var result = _sut.UpdateTheta(prior, spammerProbMean: 0.0, annotation: 1, Verdict.TrueAlarm, learningRate: 1.0);

        result.Alpha.Should().BeApproximately(1.3, precision: 1e-10);
        result.Beta.Should().Be(9.0);
    }

    // ── UpdateTheta: FALSE_ALARM + flagged threat ─────────────────────────────

    [Fact]
    public void UpdateTheta_FalseAlarm_FlaggedThreat_IncreasesAlpha()
    {
        var prior = new BetaParameters(1.0, 9.0);
        const double sp = 0.8, lr = 0.5;

        var result = _sut.UpdateTheta(prior, sp, annotation: 3, Verdict.FalseAlarm, lr);

        result.Alpha.Should().BeApproximately(prior.Alpha + lr * sp, precision: 1e-10);
    }

    [Fact]
    public void UpdateTheta_FalseAlarm_FlaggedThreat_BetaUnchanged()
    {
        var prior = new BetaParameters(1.0, 9.0);

        var result = _sut.UpdateTheta(prior, spammerProbMean: 0.8, annotation: 3, Verdict.FalseAlarm);

        result.Beta.Should().Be(prior.Beta);
    }

    // ── UpdateTheta: FALSE_ALARM + NOT flagged ────────────────────────────────

    [Fact]
    public void UpdateTheta_FalseAlarm_NotFlagged_IncreasesBeta()
    {
        var prior = new BetaParameters(1.0, 9.0);
        const double lr = 0.5;

        var result = _sut.UpdateTheta(prior, spammerProbMean: 0.1, annotation: 0, Verdict.FalseAlarm, lr);

        result.Beta.Should().BeApproximately(prior.Beta + lr * 0.3, precision: 1e-10);
    }

    [Fact]
    public void UpdateTheta_FalseAlarm_NotFlagged_AlphaUnchanged()
    {
        var prior = new BetaParameters(1.0, 9.0);

        var result = _sut.UpdateTheta(prior, spammerProbMean: 0.1, annotation: 0, Verdict.FalseAlarm);

        result.Alpha.Should().Be(prior.Alpha);
    }

    // ── UpdateTheta: learning-rate scaling ────────────────────────────────────

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void UpdateTheta_LearningRate_ScalesDeltaProportionally(double lr)
    {
        var prior = new BetaParameters(1.0, 9.0);
        const double sp = 0.2;

        var result = _sut.UpdateTheta(prior, sp, annotation: 3, Verdict.TrueAlarm, lr);

        result.Beta.Should().BeApproximately(prior.Beta + lr * (1.0 - sp), precision: 1e-10);
    }

    [Fact]
    public void UpdateTheta_DefaultLearningRate_IsHalfPoint5()
    {
        // Calling without explicit lr must give the same result as lr=0.5
        var prior = new BetaParameters(1.0, 9.0);
        const double sp = 0.2;

        var defaultResult  = _sut.UpdateTheta(prior, sp, annotation: 3, Verdict.TrueAlarm);
        var explicitResult = _sut.UpdateTheta(prior, sp, annotation: 3, Verdict.TrueAlarm, learningRate: 0.5);

        defaultResult.Should().Be(explicitResult);
    }
}
