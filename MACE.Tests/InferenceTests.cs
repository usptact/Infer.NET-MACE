using MACE;

namespace MACE.Tests
{
    public class InferenceTests
    {
        /// <summary>
        /// The shipped sample against its shipped ground truth. Items 6 and 8 are cases where the
        /// annotations genuinely favour a different label than the truth file does, so 8/10 is the
        /// ceiling for this model on this data, not a defect. This pins accuracy against silent
        /// regressions in the model or the sparse encoding.
        /// </summary>
        [Fact]
        public void SampleData_ReproducesKnownAccuracyAgainstGroundTruth()
        {
            var run = TestSupport.Run("sample_data.txt");
            var predicted = TestSupport.ArgmaxLabels(run.Posterior, run.NumItems);

            var truth = File.ReadAllLines("true_labels.txt")
                .Where(line => !string.IsNullOrWhiteSpace(line))
                .Select(line => int.Parse(line.Trim()))
                .ToArray();

            Assert.Equal(truth.Length, predicted.Length);

            int correct = predicted.Where((label, item) => label == truth[item]).Count();
            Assert.Equal(8, correct);
        }

        [Fact]
        public void SampleData_ProducesNormalisedPosteriorPerItem()
        {
            var run = TestSupport.Run("sample_data.txt");

            for (int item = 0; item < run.NumItems; item++)
            {
                var probs = run.Posterior.TDist[item].GetProbs();
                Assert.Equal(run.NumCategories, probs.Count);
                Assert.Equal(1.0, probs.Sum(), precision: 6);
            }
        }

        /// <summary>
        /// SDist[item][k] is indexed by annotation slot, not by worker. Every slot must line up
        /// with the matching entry of WorkerIndices, which is what the CSV writer relies on to
        /// map a posterior back to a worker.
        /// </summary>
        [Fact]
        public void SpammerPosterior_IsParallelToAnnotationSlots()
        {
            var run = TestSupport.Run("sample_data.txt");

            Assert.Equal(run.Annotations.WorkerIndices.Length, run.Posterior.SDist.Length);

            for (int item = 0; item < run.NumItems; item++)
            {
                Assert.Equal(run.Annotations.WorkerIndices[item].Length, run.Posterior.SDist[item].Length);

                foreach (var worker in run.Annotations.WorkerIndices[item])
                {
                    Assert.InRange(worker, 0, run.NumWorkers - 1);
                }

                foreach (var spammer in run.Posterior.SDist[item])
                {
                    Assert.InRange(spammer.GetProbTrue(), 0.0, 1.0);
                }
            }
        }

        [Fact]
        public void WorkerPosteriors_AreReturnedForEveryWorker()
        {
            var run = TestSupport.Run("sample_data.txt");

            Assert.Equal(run.NumWorkers, run.Posterior.ThetaDist.Length);
            Assert.Equal(run.NumWorkers, run.Posterior.PhiDist.Length);

            foreach (var theta in run.Posterior.ThetaDist)
            {
                Assert.InRange(theta.GetMean(), 0.0, 1.0);
            }

            foreach (var phi in run.Posterior.PhiDist)
            {
                Assert.Equal(run.NumCategories, phi.Dimension);
            }
        }

        [Fact]
        public void SameSeed_ProducesIdenticalPosteriors()
        {
            var first = TestSupport.Run("sample_data.txt", seed: 7);
            var second = TestSupport.Run("sample_data.txt", seed: 7);

            for (int item = 0; item < first.NumItems; item++)
            {
                var a = first.Posterior.TDist[item].GetProbs();
                var b = second.Posterior.TDist[item].GetProbs();

                for (int category = 0; category < a.Count; category++)
                {
                    Assert.Equal(a[category], b[category], precision: 12);
                }
            }
        }

        /// <summary>
        /// sample_data.txt converges to a single fixed point, so the seed cannot change its result.
        /// This is deliberately pinned: mistaking it for a broken --seed flag is an easy error.
        /// Seed_AffectsPosterior_WithinASingleProcess is the test that actually exercises the flag.
        /// </summary>
        [Fact]
        public void UnimodalData_IsSeedInvariant()
        {
            var baseline = TestSupport.ArgmaxLabels(
                TestSupport.Run("sample_data.txt", seed: 1).Posterior, 10);

            foreach (int seed in new[] { 7, 42, 999 })
            {
                var other = TestSupport.ArgmaxLabels(
                    TestSupport.Run("sample_data.txt", seed: seed).Posterior, 10);
                Assert.Equal(baseline, other);
            }
        }

        /// <summary>
        /// Regression test for initialisation being compiled into the generated algorithm as a
        /// constant. When that happened, the cached algorithm kept the first run's values and every
        /// later run in the same process reused them, so the seed appeared to work from the command
        /// line (one run per process) while doing nothing in a loop. Several seeds in a single
        /// process must not all produce the same posterior.
        /// </summary>
        [Fact]
        public void Seed_AffectsPosterior_WithinASingleProcess()
        {
            using var corpus = TestSupport.PerfectlyTiedCorpus();

            var posteriors = new HashSet<string>();
            foreach (int seed in Enumerable.Range(1, 6))
            {
                var run = TestSupport.Run(corpus.Path, seed: seed);
                var probs = run.Posterior.TDist[0].GetProbs();
                posteriors.Add(string.Join(",", probs.Select(p => p.ToString("G17"))));
            }

            Assert.True(
                posteriors.Count > 1,
                "Every seed produced an identical posterior in one process, which means the "
                + "initialisation is not reaching inference and --seed has become a no-op.");
        }

        /// <summary>
        /// MACE cannot break a perfectly balanced disagreement, and does not pretend to: the item
        /// posterior lands on an exact tie and every worker gets the same competence. The label the
        /// CSV reports for such an item is a coin flip, not a finding.
        /// </summary>
        [Fact]
        public void PerfectlyTiedData_YieldsTiedPosteriorAndEqualCompetence()
        {
            using var corpus = TestSupport.PerfectlyTiedCorpus();
            var run = TestSupport.Run(corpus.Path, seed: 1);

            for (int item = 0; item < run.NumItems; item++)
            {
                var probs = run.Posterior.TDist[item].GetProbs();
                Assert.Equal(0.5, probs[0], precision: 6);
                Assert.Equal(0.5, probs[1], precision: 6);
            }

            var competence = run.Posterior.ThetaDist.Select(t => t.GetMean()).ToArray();
            Assert.Equal(4, competence.Length);
            foreach (var theta in competence)
            {
                Assert.Equal(competence[0], theta, precision: 6);
            }
        }

        /// <summary>
        /// Whichever side the tie falls on, it must fall the same way for every item: the workers
        /// are perfectly correlated, so a per-item split would mean inference had lost the
        /// structure entirely.
        /// </summary>
        [Fact]
        public void PerfectlyTiedData_ResolvesConsistentlyAcrossItems()
        {
            using var corpus = TestSupport.PerfectlyTiedCorpus();
            var run = TestSupport.Run(corpus.Path, seed: 1);

            Assert.Single(TestSupport.ArgmaxLabels(run.Posterior, run.NumItems).Distinct());
        }
    }
}
