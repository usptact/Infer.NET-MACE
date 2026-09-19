using MACE;
using Microsoft.ML.Probabilistic.Distributions;

namespace MACE.Tests
{
    public class ModelPriorsIoTests : IDisposable
    {
        private readonly string _path = Path.Combine(
            Path.GetTempPath(), $"mace_priors_{Guid.NewGuid():N}.txt");

        public void Dispose()
        {
            try
            {
                File.Delete(_path);
            }
            catch (IOException)
            {
                // Not worth failing a test over.
            }
        }

        [Fact]
        public void RoundTrip_PreservesWorkerParameters()
        {
            var original = new ModelPriors(
                ThetaDist: new[] { new Beta(2.5, 7.5), new Beta(1.25, 3.75) },
                PhiDist: new[]
                {
                    new Dirichlet(1.5, 2.5, 3.5),
                    new Dirichlet(4.25, 0.75, 2.0)
                });

            ModelPriorsIo.Save(original, _path);
            var loaded = ModelPriorsIo.Load(_path, expectedWorkers: 2, expectedCategories: 3);

            for (int worker = 0; worker < 2; worker++)
            {
                Assert.Equal(original.ThetaDist[worker].TrueCount, loaded.ThetaDist[worker].TrueCount, precision: 12);
                Assert.Equal(original.ThetaDist[worker].FalseCount, loaded.ThetaDist[worker].FalseCount, precision: 12);

                var before = original.PhiDist[worker].PseudoCount;
                var after = loaded.PhiDist[worker].PseudoCount;
                for (int category = 0; category < before.Count; category++)
                {
                    Assert.Equal(before[category], after[category], precision: 12);
                }
            }
        }

        /// <summary>
        /// A posterior is a valid set of priors; persisting one is what makes the incremental path
        /// usable across processes rather than only within a single run.
        /// </summary>
        [Fact]
        public void PosteriorCanBeSavedAndReloadedAsPriors()
        {
            var run = TestSupport.Run("sample_data.txt");

            ModelPriorsIo.Save(run.Posterior, _path);
            var reloaded = ModelPriorsIo.Load(_path, run.NumWorkers, run.NumCategories);

            Assert.Equal(run.NumWorkers, reloaded.ThetaDist.Length);
            Assert.Equal(run.NumWorkers, reloaded.PhiDist.Length);

            for (int worker = 0; worker < run.NumWorkers; worker++)
            {
                Assert.Equal(
                    run.Posterior.ThetaDist[worker].GetMean(),
                    reloaded.ThetaDist[worker].GetMean(),
                    precision: 12);
            }
        }

        /// <summary>
        /// Loading priors that describe a different worker set would attribute one worker's history
        /// to another, so the mismatch is an error rather than something to paper over.
        /// </summary>
        [Fact]
        public void Load_WorkerCountMismatch_Throws()
        {
            var priors = TestSupport.UniformPriors(numWorkers: 3, numCategories: 2);
            ModelPriorsIo.Save(priors, _path);

            var ex = Assert.Throws<InvalidOperationException>(
                () => ModelPriorsIo.Load(_path, expectedWorkers: 5, expectedCategories: 2));
            Assert.Contains("describes 3 workers but the data has 5", ex.Message);
        }

        [Fact]
        public void Load_CategoryCountMismatch_Throws()
        {
            var priors = TestSupport.UniformPriors(numWorkers: 3, numCategories: 2);
            ModelPriorsIo.Save(priors, _path);

            var ex = Assert.Throws<InvalidOperationException>(
                () => ModelPriorsIo.Load(_path, expectedWorkers: 3, expectedCategories: 4));
            Assert.Contains("describes 2 categories", ex.Message);
        }

        [Fact]
        public void Load_MissingFile_Throws()
        {
            var ex = Assert.Throws<InvalidOperationException>(
                () => ModelPriorsIo.Load(_path, expectedWorkers: 1, expectedCategories: 2));
            Assert.Contains("does not exist", ex.Message);
        }

        [Fact]
        public void Load_MissingHeaderLine_Throws()
        {
            File.WriteAllLines(_path, new[] { "# MACE priors v1", "categories,2", "theta,1,1" });

            var ex = Assert.Throws<InvalidOperationException>(
                () => ModelPriorsIo.Load(_path, expectedWorkers: 1, expectedCategories: 2));
            Assert.Contains("missing its 'workers' line", ex.Message);
        }

        [Fact]
        public void Load_TruncatedPhiRow_Throws()
        {
            File.WriteAllLines(_path, new[]
            {
                "# MACE priors v1",
                "workers,1",
                "categories,3",
                "theta,1,1",
                "phi,1,1"
            });

            var ex = Assert.Throws<InvalidOperationException>(
                () => ModelPriorsIo.Load(_path, expectedWorkers: 1, expectedCategories: 3));
            Assert.Contains("Malformed phi line", ex.Message);
        }

        [Fact]
        public void Load_NonNumericValue_Throws()
        {
            File.WriteAllLines(_path, new[]
            {
                "# MACE priors v1",
                "workers,1",
                "categories,2",
                "theta,1,notanumber",
                "phi,1,1"
            });

            Assert.Throws<InvalidOperationException>(
                () => ModelPriorsIo.Load(_path, expectedWorkers: 1, expectedCategories: 2));
        }

        [Fact]
        public void Load_DeclaredWorkerCountNotMatchingRows_Throws()
        {
            File.WriteAllLines(_path, new[]
            {
                "# MACE priors v1",
                "workers,2",
                "categories,2",
                "theta,1,1",
                "phi,1,1"
            });

            var ex = Assert.Throws<InvalidOperationException>(
                () => ModelPriorsIo.Load(_path, expectedWorkers: 2, expectedCategories: 2));
            Assert.Contains("declares 2 workers but contains 1 theta", ex.Message);
        }

        /// <summary>
        /// Carrying priors from one batch into a second batch of different annotations must run and
        /// produce a usable posterior. This is the path ModelPosterior : ModelPriors exists for.
        /// </summary>
        [Fact]
        public void PriorsCarriedIntoASecondBatch_ProduceAUsablePosterior()
        {
            using var firstBatch = new TempCsv("w1,w2,w3", "0,1,0", "1,1,0", "0,0,1");
            using var secondBatch = new TempCsv("w1,w2,w3", "1,1,1", "0,0,0", "1,0,1");

            var first = TestSupport.Run(firstBatch.Path);
            ModelPriorsIo.Save(first.Posterior, _path);

            var carried = ModelPriorsIo.Load(_path, first.NumWorkers, first.NumCategories);

            using var reader = new CsvReader(secondBatch.Path);
            reader.Read();
            var trainer = new MACETrain(reader.GetNumWorkers(), reader.GetNumItems(), reader.GetNumCategories(), 50, 42);
            var posterior = trainer.InferModelData(reader.GetSparseData(), carried);

            Assert.Equal(reader.GetNumItems(), posterior.TDist.Length);
            foreach (var theta in posterior.ThetaDist)
            {
                Assert.InRange(theta.GetMean(), 0.0, 1.0);
            }
        }
    }
}
