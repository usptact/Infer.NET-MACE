using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace MACE
{
    /// <summary>
    /// Main program class for the MACE (Multi-Annotator Competence Estimation) algorithm.
    /// This program implements the MACE algorithm described in "Learning Whom to Trust with MACE" 
    /// by Dirk Hovy et al, NAACL 2013.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// Main entry point of the application.
        /// </summary>
        /// <param name="args">Command line arguments. The first argument should be the path to the CSV data file.</param>
        /// <returns>Exit code: 0 for success, 1 for error.</returns>
        public static int Main(string[] args)
        {
            try
            {
                if (args.Length < 1)
                {
                    Console.WriteLine("Usage: MACE.exe <CSV_FILE>");
                    Console.WriteLine();
                    Console.WriteLine("Example: MACE.exe sample_data.txt");
                    Console.WriteLine();
                    Console.WriteLine("Output files:");
                    Console.WriteLine("  - item_labels.csv: Inferred label probabilities for each item");
                    Console.WriteLine("  - worker_spammer_probs.csv: Spammer probabilities for each worker-item pair");
                    return 1;
                }

                string fileName = args[0];

                if (!File.Exists(fileName))
                {
                    Console.WriteLine($"ERROR: The file '{fileName}' does not exist!");
                    return 1;
                }

                Console.WriteLine("=== MACE: Multi-Annotator Competence Estimation ===");
                Console.WriteLine();

                // Generate output file names based on input file
                string baseName = Path.GetFileNameWithoutExtension(fileName);
                string outputDir = Path.GetDirectoryName(fileName) ?? ".";
                string itemLabelsFile = Path.Combine(outputDir, $"{baseName}_item_labels.csv");
                string spammerProbsFile = Path.Combine(outputDir, $"{baseName}_worker_spammer_probs.csv");

                Console.WriteLine($"Input file: {fileName}");
                Console.WriteLine($"Output files:");
                Console.WriteLine($"  Item labels: {itemLabelsFile}");
                Console.WriteLine($"  Spammer probabilities: {spammerProbsFile}");
                Console.WriteLine();

                // Read and validate data
                Console.WriteLine("Reading input data...");
                using var reader = new CsvReader(fileName);
                reader.Read();
                
                var data = reader.GetData();
                int numWorkers = reader.GetNumWorkers();
                int numItems = reader.GetNumItems();
                int numCategories = reader.GetNumCategories() + 1;

                Console.WriteLine("*** DATA STATISTICS ***");
                Console.WriteLine($"Number of items: {numItems}");
                Console.WriteLine($"Number of workers: {numWorkers}");
                Console.WriteLine($"Number of categories: {numCategories}");
                Console.WriteLine();

                // Display data quality validation results
                var validationMessages = reader.GetValidationMessages();
                if (validationMessages.Count > 0)
                {
                    Console.WriteLine("*** DATA QUALITY VALIDATION ***");
                    foreach (var message in validationMessages)
                    {
                        Console.WriteLine(message);
                    }
                    Console.WriteLine();
                }

                // Validate data dimensions
                if (numWorkers <= 0 || numItems <= 0 || numCategories <= 0)
                {
                    Console.WriteLine("ERROR: Invalid data dimensions. Please check your CSV file.");
                    return 1;
                }

                // Initialize MACE model priors
                Console.WriteLine("Initializing MACE model priors...");
                var initPriors = InitializePriors(numWorkers, numCategories);

                // Create and train the MACE model
                Console.WriteLine("Creating probabilistic model...");
                var trainer = new MACETrain(numWorkers, numItems, numCategories);
                trainer.CreateModel();
                trainer.InitializeLabels(numItems, numCategories);
                trainer.SetModelData(initPriors);

                Console.WriteLine("Running probabilistic inference...");
                var posterior = trainer.InferModelData(data);

                // Write results to CSV files
                Console.WriteLine("Writing results to CSV files...");
                WriteItemLabelsToCsv(posterior, numItems, numCategories, itemLabelsFile);
                WriteSpammerProbabilitiesToCsv(posterior, numItems, numWorkers, spammerProbsFile);

                Console.WriteLine();
                Console.WriteLine("*** INFERENCE COMPLETED SUCCESSFULLY ***");
                Console.WriteLine($"Results written to:");
                Console.WriteLine($"  - {itemLabelsFile}");
                Console.WriteLine($"  - {spammerProbsFile}");

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: An unexpected error occurred: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                return 1;
            }
        }

        /// <summary>
        /// Initializes the prior distributions for the MACE model.
        /// </summary>
        /// <param name="numWorkers">Number of workers in the dataset.</param>
        /// <param name="numCategories">Number of label categories.</param>
        /// <returns>ModelData containing initialized priors.</returns>
        private static ModelData InitializePriors(int numWorkers, int numCategories)
        {
            var initPriors = new ModelData();

            // Beta(1,1) priors for spammer probabilities (theta) - uniform prior
            var thetaDist = new Beta[numWorkers];
            for (int i = 0; i < numWorkers; i++)
            {
                thetaDist[i] = new Beta(1, 1);
            }

            // Dirichlet(1,...,1) priors for spammer label preferences (phi) - uniform prior
            var phiDist = new Dirichlet[numWorkers];
            var uniformConcentration = Enumerable.Repeat(1.0, numCategories).ToArray();
            for (int i = 0; i < numWorkers; i++)
            {
                phiDist[i] = new Dirichlet(uniformConcentration);
            }

            initPriors.ThetaDist = thetaDist;
            initPriors.PhiDist = phiDist;

            return initPriors;
        }

        /// <summary>
        /// Writes the inferred item label probabilities to a CSV file.
        /// </summary>
        /// <param name="posterior">Posterior distributions from MACE inference.</param>
        /// <param name="numItems">Number of items in the dataset.</param>
        /// <param name="numCategories">Number of label categories.</param>
        /// <param name="outputFile">Path to the output CSV file.</param>
        private static void WriteItemLabelsToCsv(ModelData posterior, int numItems, int numCategories, string outputFile)
        {
            try
            {
                using var writer = new StreamWriter(outputFile);
                
                // Write header
                var headerColumns = new List<string> { "Item" };
                for (int category = 0; category < numCategories; category++)
                {
                    headerColumns.Add($"Label_{category}_Probability");
                }
                headerColumns.Add("Most_Probable_Label");
                headerColumns.Add("Confidence");
                
                writer.WriteLine(string.Join(",", headerColumns));

                // Write data rows
                for (int item = 0; item < numItems; item++)
                {
                    var labelDist = posterior.TDist[item];
                    var probs = labelDist.GetProbs();
                    
                    var row = new List<string> { $"Item_{item + 1}" };
                    
                    // Add probabilities for each category
                    for (int category = 0; category < numCategories; category++)
                    {
                        row.Add($"{probs[category]:F6}");
                    }
                    
                    // Find most probable label and confidence
                    double maxProb = double.MinValue;
                    int mostProbableLabel = 0;
                    for (int i = 0; i < probs.Count; i++)
                    {
                        if (probs[i] > maxProb)
                        {
                            maxProb = probs[i];
                            mostProbableLabel = i;
                        }
                    }
                    double confidence = maxProb;
                    
                    row.Add($"Label_{mostProbableLabel}");
                    row.Add($"{confidence:F6}");
                    
                    writer.WriteLine(string.Join(",", row));
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to write item labels to '{outputFile}': {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Writes the worker spammer probabilities to a CSV file.
        /// </summary>
        /// <param name="posterior">Posterior distributions from MACE inference.</param>
        /// <param name="numItems">Number of items in the dataset.</param>
        /// <param name="numWorkers">Number of workers in the dataset.</param>
        /// <param name="outputFile">Path to the output CSV file.</param>
        private static void WriteSpammerProbabilitiesToCsv(ModelData posterior, int numItems, int numWorkers, string outputFile)
        {
            try
            {
                using var writer = new StreamWriter(outputFile);
                
                // Write header
                var headerColumns = new List<string> { "Item", "Worker", "Spammer_Probability" };
                writer.WriteLine(string.Join(",", headerColumns));

                // Write data rows
                for (int item = 0; item < numItems; item++)
                {
                    for (int worker = 0; worker < numWorkers; worker++)
                    {
                        var spamProb = posterior.SDist[item][worker].GetProbTrue();
                        
                        var row = new List<string>
                        {
                            $"Item_{item + 1}",
                            $"Worker_{worker + 1}",
                            $"{spamProb:F6}"
                        };
                        
                        writer.WriteLine(string.Join(",", row));
                    }
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to write spammer probabilities to '{outputFile}': {ex.Message}", ex);
            }
        }
    }
}
