using System.Globalization;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace MACE
{
    /// <summary>
    /// Reads and writes <see cref="ModelPriors"/> as a small text file.
    /// </summary>
    /// <remarks>
    /// <see cref="ModelPosterior"/> derives from <see cref="ModelPriors"/> so that one run's output can
    /// seed the next, but that is only useful if the worker parameters survive the end of the process.
    /// This is the file format that makes the incremental path reachable from the command line:
    /// run a batch with <c>--save-priors</c>, then start the next batch with <c>--load-priors</c>.
    ///
    /// Carrying priors forward is ordinary Bayesian updating only when the batches contain different
    /// annotations. Re-running the same annotations against their own posterior counts that evidence
    /// twice and will report false confidence.
    /// </remarks>
    public static class ModelPriorsIo
    {
        private const string Header = "# MACE priors v1";

        /// <summary>Writes worker parameter distributions to a file.</summary>
        /// <param name="priors">Distributions to persist; a <see cref="ModelPosterior"/> is accepted.</param>
        /// <param name="path">Destination file path.</param>
        /// <exception cref="ArgumentNullException">Thrown when priors or path is null.</exception>
        public static void Save(ModelPriors priors, string path)
        {
            ArgumentNullException.ThrowIfNull(priors);
            ArgumentNullException.ThrowIfNull(path);

            using var writer = new StreamWriter(path);
            writer.WriteLine(Header);
            writer.WriteLine($"workers,{priors.ThetaDist.Length}");

            int numCategories = priors.PhiDist.Length > 0 ? priors.PhiDist[0].Dimension : 0;
            writer.WriteLine($"categories,{numCategories}");

            foreach (var theta in priors.ThetaDist)
            {
                writer.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "theta,{0:R},{1:R}",
                    theta.TrueCount,
                    theta.FalseCount));
            }

            foreach (var phi in priors.PhiDist)
            {
                var counts = phi.PseudoCount.Select(c => c.ToString("R", CultureInfo.InvariantCulture));
                writer.WriteLine($"phi,{string.Join(",", counts)}");
            }
        }

        /// <summary>Reads worker parameter distributions written by <see cref="Save"/>.</summary>
        /// <param name="path">File to read.</param>
        /// <param name="expectedWorkers">Worker count the current dataset requires.</param>
        /// <param name="expectedCategories">Category count the current dataset requires.</param>
        /// <returns>The persisted priors.</returns>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the file is malformed, or describes a different number of workers or categories
        /// than the current dataset. Silently reusing mismatched priors would attribute one worker's
        /// history to another.
        /// </exception>
        public static ModelPriors Load(string path, int expectedWorkers, int expectedCategories)
        {
            ArgumentNullException.ThrowIfNull(path);

            if (!File.Exists(path))
            {
                throw new InvalidOperationException($"Priors file '{path}' does not exist.");
            }

            var lines = File.ReadAllLines(path)
                .Where(line => !string.IsNullOrWhiteSpace(line) && !line.StartsWith('#'))
                .ToArray();

            int workers = ReadScalar(lines, "workers", path);
            int categories = ReadScalar(lines, "categories", path);

            if (workers != expectedWorkers)
            {
                throw new InvalidOperationException(
                    $"Priors file '{path}' describes {workers} workers but the data has {expectedWorkers}. "
                    + "Priors can only be carried forward between batches that share the same worker columns.");
            }

            if (categories != expectedCategories)
            {
                throw new InvalidOperationException(
                    $"Priors file '{path}' describes {categories} categories but the data has {expectedCategories}.");
            }

            var thetas = new List<Beta>();
            var phis = new List<Dirichlet>();

            foreach (var line in lines)
            {
                var fields = line.Split(',');

                if (fields[0] == "theta")
                {
                    if (fields.Length != 3)
                    {
                        throw new InvalidOperationException(
                            $"Malformed theta line in '{path}': expected 2 values, found {fields.Length - 1}.");
                    }

                    thetas.Add(new Beta(ParseDouble(fields[1], path), ParseDouble(fields[2], path)));
                }
                else if (fields[0] == "phi")
                {
                    if (fields.Length != categories + 1)
                    {
                        throw new InvalidOperationException(
                            $"Malformed phi line in '{path}': expected {categories} values, found {fields.Length - 1}.");
                    }

                    var counts = Vector.Zero(categories);
                    for (int category = 0; category < categories; category++)
                    {
                        counts[category] = ParseDouble(fields[category + 1], path);
                    }

                    phis.Add(new Dirichlet(counts));
                }
            }

            if (thetas.Count != workers || phis.Count != workers)
            {
                throw new InvalidOperationException(
                    $"Priors file '{path}' declares {workers} workers but contains {thetas.Count} theta "
                    + $"and {phis.Count} phi entries.");
            }

            return new ModelPriors(thetas.ToArray(), phis.ToArray());
        }

        private static int ReadScalar(string[] lines, string key, string path)
        {
            var line = lines.FirstOrDefault(l => l.StartsWith(key + ",", StringComparison.Ordinal));
            if (line == null)
            {
                throw new InvalidOperationException($"Priors file '{path}' is missing its '{key}' line.");
            }

            if (!int.TryParse(line.Split(',')[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
            {
                throw new InvalidOperationException($"Priors file '{path}' has a non-numeric '{key}' value.");
            }

            return value;
        }

        private static double ParseDouble(string field, string path)
        {
            if (!double.TryParse(field, NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
            {
                throw new InvalidOperationException($"Priors file '{path}' contains a non-numeric value '{field}'.");
            }

            return value;
        }
    }
}
