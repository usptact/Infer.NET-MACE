using FluentAssertions;
using MACE.Services;
using Xunit;

namespace MACE.Tests.Services;

public class PriorUpdateServiceTests
{
    private readonly PriorUpdateService _sut = new();

    private static double[] Uniform(int k = 5) => Enumerable.Repeat(1.0, k).ToArray();

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

    // ── UpdateBeliefs: absent sensor (reading == -1) ──────────────────────────

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void UpdateBeliefs_AbsentSensor_ReturnsUnchangedThetaAndPhi(int trueLevel)
    {
        var theta = new BetaParameters(2.0, 8.0);
        var phi   = Uniform();

        var result = _sut.UpdateBeliefs(theta, phi, reading: -1, trueLevel);

        result.Theta.Should().Be(theta);
        result.Phi.Should().Equal(phi);
    }

    // ── UpdateBeliefs: reading disagrees with gold (reading != trueLevel) ──────
    // A reliable sensor must report the truth, so a mismatch means the sensor was
    // certainly faulty: responsibility r = 1.

    [Fact]
    public void UpdateBeliefs_DisagreesWithGold_ResponsibilityIsOne_ShiftsAlphaAndPhi()
    {
        var theta = new BetaParameters(1.0, 9.0);
        var phi   = Uniform();
        const double lr = 0.5;

        var result = _sut.UpdateBeliefs(theta, phi, reading: 3, trueLevel: 0, lr);

        // r = 1 ⇒ alpha += lr, beta unchanged
        result.Theta.Alpha.Should().BeApproximately(1.0 + lr, precision: 1e-10);
        result.Theta.Beta.Should().Be(9.0);
        // phi bucket for the emitted reading gains lr, all others unchanged
        result.Phi[3].Should().BeApproximately(1.0 + lr, precision: 1e-10);
        result.Phi.Where((_, i) => i != 3).Should().AllSatisfy(c => c.Should().Be(1.0));
    }

    [Fact]
    public void UpdateBeliefs_DisagreesWithGold_PenaltyIndependentOfPriorTrust()
    {
        // A highly trusted sensor (θ̄ ≈ 0.01) that disagrees with gold is fully
        // penalised — the update no longer discounts surprising evidence (Issue 2).
        var trusted = new BetaParameters(1.0, 99.0);
        var phi     = Uniform();

        var result = _sut.UpdateBeliefs(trusted, phi, reading: 4, trueLevel: 0, learningRate: 1.0);

        result.Theta.Alpha.Should().BeApproximately(2.0, precision: 1e-10); // += lr*1
        result.Theta.Beta.Should().Be(99.0);
        result.Phi[4].Should().BeApproximately(2.0, precision: 1e-10);
    }

    // ── UpdateBeliefs: reading matches gold (reading == trueLevel) ─────────────
    // The reading is consistent with either a reliable sensor or a faulty one that
    // happened to emit the truth: r = θ̄·φ̄[a] / (θ̄·φ̄[a] + (1 − θ̄)).

    [Fact]
    public void UpdateBeliefs_MatchesGold_UsesFaultResponsibility()
    {
        // θ̄ = 4/5 = 0.8;  φ̄[0] = 1/4 = 0.25
        // faulty = 0.8*0.25 = 0.2;  reliable = 0.2;  r = 0.2/0.4 = 0.5
        var theta = new BetaParameters(4.0, 1.0);
        var phi   = new[] { 1.0, 3.0 };
        const double lr = 0.5;

        var result = _sut.UpdateBeliefs(theta, phi, reading: 0, trueLevel: 0, lr);

        result.Theta.Alpha.Should().BeApproximately(4.0 + lr * 0.5, precision: 1e-10); // 4.25
        result.Theta.Beta.Should().BeApproximately(1.0 + lr * 0.5, precision: 1e-10);  // 1.25
        result.Phi[0].Should().BeApproximately(1.0 + lr * 0.5, precision: 1e-10);      // 1.25
        result.Phi[1].Should().Be(3.0);                                                // unchanged
    }

    [Fact]
    public void UpdateBeliefs_MatchesGold_MoreTrustedSensorGetsSmallerFaultCredit()
    {
        // Same reading/gold, but a more reliable prior (lower θ̄) yields a smaller r,
        // so alpha grows less. This is the correct "a right reading is less
        // informative when we already trust the sensor" behaviour.
        var phi = Uniform();

        var trusted   = _sut.UpdateBeliefs(new BetaParameters(1.0, 99.0), phi, reading: 2, trueLevel: 2);
        var untrusted = _sut.UpdateBeliefs(new BetaParameters(50.0, 50.0), phi, reading: 2, trueLevel: 2);

        double trustedDelta   = trusted.Theta.Alpha   - 1.0;
        double untrustedDelta = untrusted.Theta.Alpha - 50.0;

        trustedDelta.Should().BeLessThan(untrustedDelta);
        trustedDelta.Should().BeGreaterThan(0);
    }

    // ── UpdateBeliefs: conjugate mass conservation ────────────────────────────

    [Fact]
    public void UpdateBeliefs_ThetaGainsExactlyLearningRateOfMass()
    {
        // α and β together always increase by exactly the learning rate
        // (α += lr*r, β += lr*(1-r)), for any reading/gold combination.
        var theta = new BetaParameters(3.0, 7.0);
        var phi   = Uniform();
        const double lr = 0.5;

        foreach (var (reading, gold) in new[] { (2, 2), (4, 0), (0, 1) })
        {
            var result = _sut.UpdateBeliefs(theta, phi, reading, gold, lr);
            double added = (result.Theta.Alpha - theta.Alpha) + (result.Theta.Beta - theta.Beta);
            added.Should().BeApproximately(lr, precision: 1e-10);
        }
    }

    // ── UpdateBeliefs: learning-rate scaling ──────────────────────────────────

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void UpdateBeliefs_LearningRate_ScalesDeltaProportionally(double lr)
    {
        // Disagreement ⇒ r = 1, so the alpha and phi deltas equal lr exactly.
        var theta = new BetaParameters(1.0, 9.0);
        var phi   = Uniform();

        var result = _sut.UpdateBeliefs(theta, phi, reading: 3, trueLevel: 0, lr);

        result.Theta.Alpha.Should().BeApproximately(1.0 + lr, precision: 1e-10);
        result.Phi[3].Should().BeApproximately(1.0 + lr, precision: 1e-10);
    }

    [Fact]
    public void UpdateBeliefs_DefaultLearningRate_IsHalfPoint5()
    {
        var theta = new BetaParameters(1.0, 9.0);

        var defaultResult  = _sut.UpdateBeliefs(theta, Uniform(), reading: 3, trueLevel: 0);
        var explicitResult = _sut.UpdateBeliefs(theta, Uniform(), reading: 3, trueLevel: 0, learningRate: 0.5);

        defaultResult.Theta.Should().Be(explicitResult.Theta);
        defaultResult.Phi.Should().Equal(explicitResult.Phi);
    }

    // ── UpdateBeliefs: input immutability & guards ────────────────────────────

    [Fact]
    public void UpdateBeliefs_DoesNotMutateCallerPhi()
    {
        var theta = new BetaParameters(1.0, 9.0);
        var phi   = Uniform();

        _sut.UpdateBeliefs(theta, phi, reading: 3, trueLevel: 0, learningRate: 1.0);

        phi.Should().Equal(Uniform()); // caller's array is untouched
    }

    [Theory]
    [InlineData(-2)]
    [InlineData(5)]
    public void UpdateBeliefs_ReadingOutOfRange_Throws(int reading)
    {
        var theta = new BetaParameters(1.0, 9.0);

        var act = () => _sut.UpdateBeliefs(theta, Uniform(), reading, trueLevel: 0);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
