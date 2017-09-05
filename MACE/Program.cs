using System;
using System.Linq;
using System.IO;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace MACE
{
    class Program
    {
        static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: MACE.exe <CSV>");
                return 1;
            }

            string fileName = args[0];

            if (!File.Exists(fileName))
            {
                Console.WriteLine("ERROR: The file {0} does not exist!", fileName);
                return 1;
            }

            //
            // Sample data:
            // 7 workers
            //  - 1-6 are average workers making 1-2 mistakes each
            //  - 7th is a total spammer, always puts 0
            //  - 8th is a perfect worker
            // Each item gets 4 answers (data is missing at random, indicated by "-1")
            // True label information is in "true_labels.txt"
            //
            CsvReader reader = new CsvReader(fileName);
            reader.read();
            int[][] data = reader.getData();

            int numWorkers = reader.getNumWorkers();
            int numItems = reader.getNumItems();
            int numCategories = reader.getNumCategories() + 1;

            Console.WriteLine("*** DATA STATISTICS ***");
            Console.WriteLine("Number of items: " + numItems);
            Console.WriteLine("Number of workers: " + numWorkers);
            Console.WriteLine("Number of categories: " + numCategories + "\n");

            //
            // MACE: Init priors
            //

            ModelData initPriors = new ModelData();

            Beta[] thetaDist = new Beta[numWorkers];
            Dirichlet[] phiDist = new Dirichlet[numWorkers];
            for (int i = 0; i < numWorkers; i++)
            {
                thetaDist[i] = new Beta(1, 1);
                phiDist[i] = new Dirichlet(Enumerable.Repeat<double>(1.0, numCategories).ToArray());
            }

            initPriors.thetaDist = thetaDist;
            initPriors.phiDist = phiDist;

            //
            // MACE: Instantiate & run model trainer
            //

            MACETrain trainer = new MACETrain(numWorkers, numItems, numCategories);

            trainer.CreateModel();
            trainer.InitializeLabels(numItems, numCategories);
            trainer.SetModelData(initPriors);

            ModelData posterior = trainer.InferModelData(data);

            Console.WriteLine("*** INFERRED ITEM LABELS ***");
            for (int item = 0; item < numItems; item++)
                Console.WriteLine("\tItem {0}: " + posterior.TDist[item], item);

            Console.WriteLine("\n*** IS SPAMMER ***");
            for (int worker = 0; worker < numWorkers; worker++)
            {
                Console.WriteLine("Worker #{0}", worker);
                for (int item = 0; item < numItems; item++)
                    Console.WriteLine("\tItem {0}: " + posterior.SDist[item][worker].GetProbTrue(), item);
            }

            Console.Write("\nPress any key...");
            Console.Read();

            return 0;
        }
    }
}
