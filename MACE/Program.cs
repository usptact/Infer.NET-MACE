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
            // MACE
            //

            MACETrain trainer = new MACETrain(numWorkers, numItems, numCategories);

            trainer.CreateModel();
            trainer.InitializeLabels(numItems, numCategories);

            ModelData posterior = trainer.InferModelData(data);

            Console.WriteLine("*** INFERRED ITEM LABELS ***");
            for (int item = 0; item < numItems; item++)
                Console.WriteLine("\tItem {0}: " + posterior.Tprior[item], item);

            Console.WriteLine("\n*** IS SPAMMER ***");
            for (int worker = 0; worker < numWorkers; worker++)
            {
                Console.WriteLine("Worker #{0}", worker);
                for (int item = 0; item < numItems; item++)
                {
                    Console.WriteLine("\tItem {0}: " + posterior.Sprior[item][worker].GetProbTrue(), item);
                }
            }

            return 0;

            //
            // model variables
            //

            Range n = new Range(numItems).Named("Item");
            Range m = new Range(numWorkers).Named("Worker");

            var T = Variable.Array<int>(n).Named("TrueLabels");
            var S = Variable.Array(Variable.Array<bool>(m), n).Named("IsSpammer");
            var A = Variable.Array(Variable.Array<int>(m), n).Named("Answer");

            //
            // Model parameters and their initialization
            //

            var theta = Variable.Array<double>(m).Named("trust");
            var ksi = Variable.Array<Vector>(m).Named("ksi");

            double[] initCounts = Enumerable.Repeat<double>(1.0, numCategories).ToArray();
            
            using (Variable.ForEach(m))
            {
                theta[m] = Variable.Random(new Beta(2, 2));
                ksi[m] = Variable.Random(new Dirichlet(initCounts));
            }

            //
            // Generative model
            //

            using (Variable.ForEach(n))                             // loop over items
            {
                T[n] = Variable.DiscreteUniform(numCategories);     // hidden true label
                using (Variable.ForEach(m))                         // loop over workers
                {
                    S[n][m] = Variable.Bernoulli(theta[m]);         // spammer/not spammer?
                    using (Variable.If(A[n][m] > -1))               // look only at observed data
                    {
                        using (Variable.If(S[n][m] == false))
                        {
                            A[n][m] = T[n];                         // not spammer: assign hiddern true label
                        }
                        using (Variable.If(S[n][m] == true))
                        {
                            A[n][m] = Variable.Discrete(ksi[m]);    // spammer: assign label from spammer's "preference" parameter vector
                        }
                    }
                }
            }

            // hook up the data
            A.ObservedValue = data;

			// prevent engine from trying to infer A
			A.AddAttribute(new DoNotInfer());

            //
            // Class labels -- break symmetry
            //

            Discrete[] Tinit = new Discrete[numItems];
            for (int item = 0; item < numItems; item++)
                Tinit[item] = Discrete.PointMass(Rand.Int(numCategories), numCategories);
            T.InitialiseTo(Distribution<int>.Array(Tinit));

            //
            // Inference
            //

            InferenceEngine engine = new InferenceEngine();

            Console.WriteLine("*** INFERRED ITEM LABELS ***");
            Discrete[] TMarginal = engine.Infer<Discrete[]>(T);
            for (int item = 0; item < numItems; item++)
                Console.WriteLine("\tItem {0}: " + TMarginal[item], item);

            Console.WriteLine("\n*** IS SPAMMER ***");
            Bernoulli[][] SMarginal = engine.Infer<Bernoulli[][]>(S);
            for (int worker = 0; worker < numWorkers; worker++)
            {
                Console.WriteLine("Worker #{0}", worker);
                for (int item = 0; item < numItems; item++)
                {
                    Console.WriteLine("\tItem {0}: " + SMarginal[item][worker].GetProbTrue(), item);
                }
            }

            Console.Write("\nPress any key...");
            Console.Read();

            return 0;
        }
    }
}
