using MACE;
using MACE.Online;

namespace MACE.Tests.Online
{
    public class PriorUpdateServiceTests
    {
        private readonly PriorUpdateService _service = new();

        private static double[] UniformPhi(int categories) => Enumerable.Repeat(1.0, categories).ToArray();

        /// <summary>
        /// An honest worker reports the truth by definition, so a worker who disagreed with an
        /// established true label cannot have been honest: all the evidence goes to alpha.
        /// </summary>
        [Fact]
        public void DisagreementIsAttributedEntirelyToSpamming()
        {
            var result = _service.UpdateBeliefs(
                new BetaParameters(1, 1), UniformPhi(3), label: 2, trueLabel: 0, learningRate: 1.0);

            Assert.Equal(2.0, result.Theta.Alpha, precision: 12);
            Assert.Equal(1.0, result.Theta.Beta, precision: 12);
            Assert.Equal(2.0, result.Phi[2], precision: 12);
            Assert.Equal(1.0, result.Phi[0], precision: 12);
        }

        /// <summary>
        /// Agreement is weaker evidence, because a spammer can produce the right label by chance.
        /// With Beta(1,1) and a uniform phi over 3 categories the responsibility works out at
        /// (0.5 * 1/3) / (0.5 * 1/3 + 0.5) = 0.25.
        /// </summary>
        [Fact]
        public void AgreementSplitsCreditByResponsibility()
        {
            var result = _service.UpdateBeliefs(
                new BetaParameters(1, 1), UniformPhi(3), label: 1, trueLabel: 1, learningRate: 1.0);

            Assert.Equal(1.25, result.Theta.Alpha, precision: 12);
            Assert.Equal(1.75, result.Theta.Beta, precision: 12);
            Assert.Equal(1.25, result.Phi[1], precision: 12);
            Assert.Equal(1.0, result.Phi[0], precision: 12);
            Assert.Equal(1.0, result.Phi[2], precision: 12);
        }

        [Fact]
        public void LearningRateScalesTheEvidence()
        {
            var full = _service.UpdateBeliefs(
                new BetaParameters(1, 1), UniformPhi(2), label: 1, trueLabel: 0, learningRate: 1.0);
            var damped = _service.UpdateBeliefs(
                new BetaParameters(1, 1), UniformPhi(2), label: 1, trueLabel: 0, learningRate: 0.25);

            Assert.Equal(2.0, full.Theta.Alpha, precision: 12);
            Assert.Equal(1.25, damped.Theta.Alpha, precision: 12);
        }

        /// <summary>
        /// A worker who did not annotate the item supplies no evidence. Decaying them here would let
        /// absence itself erode a worker's history.
        /// </summary>
        [Fact]
        public void AbsentWorkerIsLeftCompletelyUnchanged()
        {
            var theta = new BetaParameters(3, 7);
            var phi = new double[] { 2.0, 5.0 };

            var result = _service.UpdateBeliefs(
                theta, phi, MACETrain.MissingAnnotation, trueLabel: 0, learningRate: 1.0, retention: 0.5);

            Assert.Equal(3.0, result.Theta.Alpha, precision: 12);
            Assert.Equal(7.0, result.Theta.Beta, precision: 12);
            Assert.Equal(phi, result.Phi);
        }

        [Fact]
        public void InputsAreNotMutated()
        {
            var phi = UniformPhi(3);
            _service.UpdateBeliefs(new BetaParameters(1, 1), phi, label: 0, trueLabel: 0, learningRate: 1.0);

            Assert.Equal(UniformPhi(3), phi);
        }

        /// <summary>
        /// With nothing forgotten, evidence accumulates without limit: a worker's parameters harden
        /// and a worker whose behaviour later changes can never be re-learned. This pins the
        /// behaviour so the trade-off stays visible.
        /// </summary>
        [Fact]
        public void WithoutForgetting_EvidenceGrowsWithoutBound()
        {
            var theta = new BetaParameters(1, 1);
            var phi = UniformPhi(2);

            for (int item = 0; item < 200; item++)
            {
                var update = _service.UpdateBeliefs(theta, phi, label: 1, trueLabel: 0, learningRate: 1.0);
                theta = update.Theta;
                phi = update.Phi;
            }

            Assert.Equal(201.0, theta.Alpha, precision: 9);
        }

        /// <summary>
        /// With forgetting on, accumulated evidence settles at learningRate / (1 - retention)
        /// above the uninformative prior, which is what lets a long-running deployment keep adapting.
        /// </summary>
        [Fact]
        public void WithForgetting_EvidenceSettlesAtTheEffectiveSampleSize()
        {
            const double learningRate = 1.0;
            const double retention = 0.9;

            var theta = new BetaParameters(1, 1);
            var phi = UniformPhi(2);

            for (int item = 0; item < 500; item++)
            {
                var update = _service.UpdateBeliefs(theta, phi, label: 1, trueLabel: 0, learningRate, retention);
                theta = update.Theta;
                phi = update.Phi;
            }

            double expected = 1.0 + PriorUpdateService.EffectiveSampleSize(learningRate, retention);

            Assert.Equal(11.0, expected, precision: 9);
            Assert.Equal(expected, theta.Alpha, precision: 6);
        }

        /// <summary>
        /// Forgetting has to be able to reverse a verdict, otherwise it is only cosmetic: a worker
        /// who spammed for a long time and then reformed must be able to recover.
        /// </summary>
        [Fact]
        public void ForgettingLetsAWorkerRecoverAfterChangingBehaviour()
        {
            const double retention = 0.9;
            var theta = new BetaParameters(1, 1);
            var phi = UniformPhi(2);

            for (int item = 0; item < 100; item++)
            {
                var update = _service.UpdateBeliefs(theta, phi, label: 1, trueLabel: 0, 1.0, retention);
                theta = update.Theta;
                phi = update.Phi;
            }

            double afterSpamming = theta.Alpha / (theta.Alpha + theta.Beta);
            Assert.True(afterSpamming > 0.8, $"Expected a strong spammer reading, got {afterSpamming}.");

            for (int item = 0; item < 100; item++)
            {
                var update = _service.UpdateBeliefs(theta, phi, label: 0, trueLabel: 0, 1.0, retention);
                theta = update.Theta;
                phi = update.Phi;
            }

            double afterReforming = theta.Alpha / (theta.Alpha + theta.Beta);
            Assert.True(afterReforming < 0.5,
                $"Expected the worker to recover, but they still read as {afterReforming}.");
        }

        [Fact]
        public void EffectiveSampleSize_IsInfiniteWhenNothingIsForgotten()
        {
            Assert.True(double.IsPositiveInfinity(PriorUpdateService.EffectiveSampleSize(0.5, 1.0)));
            Assert.Equal(5.0, PriorUpdateService.EffectiveSampleSize(0.5, 0.9), precision: 9);
        }

        [Theory]
        [InlineData(0.0, 1.0)]
        [InlineData(-1.0, 1.0)]
        [InlineData(1.0, 0.0)]
        [InlineData(1.0, 1.5)]
        [InlineData(1.0, -0.5)]
        public void InvalidRates_Throw(double learningRate, double retention)
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => _service.UpdateBeliefs(
                    new BetaParameters(1, 1), UniformPhi(2), 0, 0, learningRate, retention));
        }

        [Theory]
        [InlineData(5, 0)]
        [InlineData(0, 5)]
        [InlineData(-2, 0)]
        public void LabelsOutsideTheCategoryRange_Throw(int label, int trueLabel)
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => _service.UpdateBeliefs(new BetaParameters(1, 1), UniformPhi(2), label, trueLabel));
        }

        [Fact]
        public void UpdateFromResolvedItem_MovesOnlyTheWorkersWhoAnnotated()
        {
            var priors = TestSupport.UniformPriors(numWorkers: 3, numCategories: 2);
            var annotations = new[] { 0, 1, MACETrain.MissingAnnotation };

            var updated = _service.UpdateFromResolvedItem(priors, annotations, trueLabel: 0, learningRate: 1.0);

            // Worker 2 abstained, so their parameters must be untouched.
            Assert.Equal(1.0, updated.ThetaDist[2].TrueCount, precision: 12);
            Assert.Equal(1.0, updated.ThetaDist[2].FalseCount, precision: 12);

            // Worker 1 disagreed with the truth and takes the full weight.
            Assert.Equal(2.0, updated.ThetaDist[1].TrueCount, precision: 12);

            // Worker 0 agreed, so only the responsibility share moves.
            Assert.True(updated.ThetaDist[0].TrueCount < updated.ThetaDist[1].TrueCount);

            // The originals are untouched.
            Assert.Equal(1.0, priors.ThetaDist[1].TrueCount, precision: 12);
        }

        /// <summary>
        /// The end-to-end point of the feedback loop: a worker who is repeatedly wrong should end up
        /// looking less reliable than one who is repeatedly right.
        /// </summary>
        [Fact]
        public void RepeatedResolutions_SeparateReliableFromUnreliableWorkers()
        {
            var priors = TestSupport.UniformPriors(numWorkers: 2, numCategories: 3);

            for (int item = 0; item < 40; item++)
            {
                int truth = item % 3;
                int wrong = (truth + 1) % 3;
                priors = _service.UpdateFromResolvedItem(
                    priors, new[] { truth, wrong }, truth, learningRate: 1.0, retention: 0.95);
            }

            double reliable = priors.ThetaDist[0].GetMean();
            double unreliable = priors.ThetaDist[1].GetMean();

            Assert.True(unreliable > 0.8, $"Consistently wrong worker read as {unreliable}.");
            Assert.True(reliable < 0.5, $"Consistently right worker read as {reliable}.");
        }
    }
}
