using System.Diagnostics;
using MACE;
using MACE.Online;

namespace MACE.Tests.Online
{
    public class InferOnlineTests
    {
        private const int Workers = 5;
        private const int Categories = 3;

        private static ModelPriors Priors() => TestSupport.UniformPriors(Workers, Categories);

        private static MACETrain Model(int? seed = 42)
            => MACETrain.ForOnlineInference(Workers, Categories, 50, seed);

        [Fact]
        public void UnanimousWorkers_YieldThatLabelWithHighConfidence()
        {
            var result = Model().InferOnline(new[] { 1, 1, 1, 1, 1 }, Priors());

            Assert.Equal(1, result.Label);
            Assert.True(result.Confidence > 0.9, $"Expected high confidence, got {result.Confidence}.");
            Assert.True(result.Entropy < 0.2, $"Expected low entropy, got {result.Entropy}.");
        }

        [Fact]
        public void AbsentWorkersAreExcludedFromTheResult()
        {
            var result = Model().InferOnline(
                new[] { 2, MACETrain.MissingAnnotation, 2, MACETrain.MissingAnnotation, 2 },
                Priors());

            Assert.Equal(new[] { 0, 2, 4 }, result.ContributingWorkers);
            Assert.Equal(3, result.SpammerDist.Length);
            Assert.Equal(2, result.Label);
        }

        /// <summary>
        /// The spammer posteriors are parallel to the contributing workers, not indexed by worker id.
        /// The lookup helper exists so callers never have to get that mapping right themselves.
        /// </summary>
        [Fact]
        public void SpammerProbabilityLookup_MapsBackToWorkerIds()
        {
            var result = Model().InferOnline(
                new[] { 0, MACETrain.MissingAnnotation, 0, MACETrain.MissingAnnotation, 1 },
                Priors());

            Assert.NotNull(result.SpammerProbabilityFor(0));
            Assert.NotNull(result.SpammerProbabilityFor(2));
            Assert.NotNull(result.SpammerProbabilityFor(4));
            Assert.Null(result.SpammerProbabilityFor(1));
            Assert.Null(result.SpammerProbabilityFor(3));

            Assert.Equal(result.SpammerDist[0].GetProbTrue(), result.SpammerProbabilityFor(0)!.Value, precision: 12);
        }

        /// <summary>
        /// The lone dissenter should look less reliable than the workers who agreed with each other.
        /// </summary>
        [Fact]
        public void DissentingWorker_LooksMoreLikeASpammer()
        {
            var result = Model().InferOnline(new[] { 0, 0, 0, 0, 2 }, Priors());

            double dissenter = result.SpammerProbabilityFor(4)!.Value;
            double agreeing = result.SpammerProbabilityFor(0)!.Value;

            Assert.Equal(0, result.Label);
            Assert.True(dissenter > agreeing,
                $"Dissenting worker scored {dissenter}, agreeing worker {agreeing}.");
        }

        [Fact]
        public void ItemWithNoAnnotations_ReturnsThePriorAndSaysSo()
        {
            var annotations = Enumerable.Repeat(MACETrain.MissingAnnotation, Workers).ToArray();
            var result = Model().InferOnline(annotations, Priors());

            Assert.Empty(result.ContributingWorkers);
            Assert.Empty(result.SpammerDist);

            var probs = result.LabelDist.GetProbs();
            foreach (var p in probs)
            {
                Assert.Equal(1.0 / Categories, p, precision: 9);
            }

            // Entropy is at its maximum, which is how a caller detects "no information" numerically.
            Assert.Equal(Math.Log(Categories), result.Entropy, precision: 9);
        }

        [Fact]
        public void EntropyIsLowerWhenWorkersAgree()
        {
            var agreeing = Model().InferOnline(new[] { 0, 0, 0, 0, 0 }, Priors());
            var split = Model().InferOnline(new[] { 0, 0, 1, 1, 2 }, Priors());

            Assert.True(split.Entropy > agreeing.Entropy,
                $"Split entropy {split.Entropy} should exceed agreeing entropy {agreeing.Entropy}.");
        }

        [Fact]
        public void OnlineResultMatchesTheEquivalentBatchCall()
        {
            var annotations = new[] { 0, 1, 0, MACETrain.MissingAnnotation, 0 };

            var online = Model().InferOnline(annotations, Priors());

            var batch = new MACETrain(Workers, 1, Categories, 50, 42).InferModelData(
                new SparseAnnotations(
                    new[] { new[] { 0, 1, 2, 4 } },
                    new[] { new[] { 0, 1, 0, 0 } }),
                Priors());

            var onlineProbs = online.LabelDist.GetProbs();
            var batchProbs = batch.TDist[0].GetProbs();

            for (int category = 0; category < Categories; category++)
            {
                Assert.Equal(batchProbs[category], onlineProbs[category], precision: 12);
            }
        }

        [Fact]
        public void WrongLengthAnnotationArray_Throws()
        {
            var ex = Assert.Throws<ArgumentException>(
                () => Model().InferOnline(new[] { 0, 1 }, Priors()));
            Assert.Contains("one entry per worker", ex.Message);
        }

        [Fact]
        public void LabelOutOfRange_Throws()
        {
            Assert.Throws<ArgumentException>(
                () => Model().InferOnline(new[] { 0, 1, 9, 0, 0 }, Priors()));
        }

        [Fact]
        public void NullArguments_Throw()
        {
            Assert.Throws<ArgumentNullException>(() => Model().InferOnline(null!, Priors()));
            Assert.Throws<ArgumentNullException>(() => Model().InferOnline(new[] { 0, 0, 0, 0, 0 }, null!));
        }

        /// <summary>
        /// A multi-item model cannot serve single-item calls, and says so rather than silently
        /// inferring only its first item.
        /// </summary>
        [Fact]
        public void MultiItemModel_RefusesOnlineCalls()
        {
            var batchModel = new MACETrain(Workers, 4, Categories);

            var ex = Assert.Throws<InvalidOperationException>(
                () => batchModel.InferOnline(new[] { 0, 0, 0, 0, 0 }, Priors()));
            Assert.Contains("ForOnlineInference", ex.Message);
        }

        /// <summary>
        /// The reason this API exists. Annotations reach the compiled algorithm as observed values,
        /// so only the first call compiles and the rest are cheap. If a change ever makes the
        /// annotations part of the model structure instead, every call recompiles and this fails.
        /// </summary>
        [Fact]
        public void RepeatedCallsDoNotRecompileTheModel()
        {
            var model = Model();
            var priors = Priors();

            var first = Stopwatch.StartNew();
            model.InferOnline(new[] { 0, 0, 0, 0, 0 }, priors);
            first.Stop();

            var steadyState = new List<long>();
            for (int call = 0; call < 8; call++)
            {
                // Vary the annotations: a recompile would be triggered by the data changing shape.
                var annotations = new[] { call % Categories, 0, 1, MACETrain.MissingAnnotation, call % Categories };

                var sw = Stopwatch.StartNew();
                model.InferOnline(annotations, priors);
                sw.Stop();
                steadyState.Add(sw.ElapsedMilliseconds);
            }

            steadyState.Sort();
            long median = steadyState[steadyState.Count / 2];

            // Compilation costs seconds; a warm call costs tens of milliseconds. The bound is loose
            // enough for a slow machine while still failing outright if compilation returns.
            Assert.True(median < 200,
                $"Median warm call took {median}ms (first call {first.ElapsedMilliseconds}ms); "
                + "this suggests the model is being recompiled per call.");
        }
    }
}
