using MACE;
using Microsoft.ML.Probabilistic.Distributions;

namespace MACE.Tests
{
    /// <summary>
    /// A CSV file written to a temporary directory and deleted when the test finishes.
    /// </summary>
    public sealed class TempCsv : IDisposable
    {
        public string Path { get; }

        public TempCsv(params string[] lines)
        {
            Path = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                $"mace_test_{Guid.NewGuid():N}.csv");
            File.WriteAllLines(Path, lines);
        }

        public void Dispose()
        {
            try
            {
                File.Delete(Path);
            }
            catch (IOException)
            {
                // A leaked temp file is not worth failing a test over.
            }
        }
    }

    public static class TestSupport
    {
        /// <summary>
        /// Six items on which workers 1 and 2 always say 0 and workers 3 and 4 always say 1.
        /// "Workers 1-2 are honest" and "workers 3-4 are honest" explain this data equally well,
        /// and nothing in the data favours either. Inference converges to the symmetric fixed
        /// point: every item sits at exactly 0.5 and all four workers get the same competence.
        /// The reported label is therefore decided by arithmetic noise, which makes this corpus
        /// the sharpest available probe of whether initialisation reaches inference at all.
        /// </summary>
        public static TempCsv PerfectlyTiedCorpus() => new TempCsv(
            "w1,w2,w3,w4",
            "0,0,1,1",
            "0,0,1,1",
            "0,0,1,1",
            "0,0,1,1",
            "0,0,1,1",
            "0,0,1,1");

        /// <summary>Uniform priors, matching what <c>Program</c> builds for a real run.</summary>
        public static ModelPriors UniformPriors(int numWorkers, int numCategories)
        {
            var concentration = Enumerable.Repeat(1.0, numCategories).ToArray();
            return new ModelPriors(
                ThetaDist: Enumerable.Range(0, numWorkers).Select(_ => new Beta(1, 1)).ToArray(),
                PhiDist: Enumerable.Range(0, numWorkers).Select(_ => new Dirichlet(concentration)).ToArray());
        }

        /// <summary>Reads a CSV and runs inference over it, the way <c>Program.Main</c> does.</summary>
        public static InferenceRun Run(string csvPath, int? seed = 42, int iterations = 50)
        {
            using var reader = new CsvReader(csvPath);
            reader.Read();

            var annotations = reader.GetSparseData();
            int numWorkers = reader.GetNumWorkers();
            int numItems = reader.GetNumItems();
            int numCategories = reader.GetNumCategories();

            var trainer = new MACETrain(numWorkers, numItems, numCategories, iterations, seed);
            var posterior = trainer.InferModelData(annotations, TestSupport.UniformPriors(numWorkers, numCategories));

            return new InferenceRun(posterior, annotations, numWorkers, numItems, numCategories);
        }

        /// <summary>The most probable label for each item, in item order.</summary>
        public static int[] ArgmaxLabels(ModelPosterior posterior, int numItems)
        {
            var labels = new int[numItems];
            for (int item = 0; item < numItems; item++)
            {
                var probs = posterior.TDist[item].GetProbs();
                int best = 0;
                for (int category = 1; category < probs.Count; category++)
                {
                    if (probs[category] > probs[best])
                    {
                        best = category;
                    }
                }

                labels[item] = best;
            }

            return labels;
        }
    }

    public record InferenceRun(
        ModelPosterior Posterior,
        SparseAnnotations Annotations,
        int NumWorkers,
        int NumItems,
        int NumCategories);
}
