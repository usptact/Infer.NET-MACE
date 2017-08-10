﻿using System;
using System.Linq;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace MACE
{
    class Program
    {
        static void Main(string[] args)
        {
            const int numWorkers = 8;
            const int numItems = 10;
            const int numCategories = 3;

            //
            // Sample data:
            // 7 workers
            //  - 1-6 are average workers making 1-2 mistakes each
            //  - 7th is a total spammer, always puts 0
            //  - 8th is a perfect worker
            // Each item gets 4 answers (data is missing at random, indicated by "-1")
            // Item true label is shown in comment
            //

            int[][] data = new int[numItems][];

            data[0] = new int[] { 0, -1, -1, 1, 0, -1, -1, 0 };   // 0
            data[1] = new int[] { -1, 2, -1, -1, 1, -1, -1, 0 };   // 0
            data[2] = new int[] { 1, 1, -1, -1, -1, -1, 0, 1 };   // 1
            data[3] = new int[] { -1, 1, -1, 1, -1, -1, -1, 1 };   // 1
            data[4] = new int[] { 1, -1, -1, -1, 1, 1, 0, -1 };   // 1
            data[5] = new int[] { -1, 2, 1, 2, -1, -1, -1, -1 };   // 1
            data[6] = new int[] { 0, 0, 2, -1, -1, -1, 0, -1 };   // 0
            data[7] = new int[] { 1, -1, -1, -1, 0, 0, 0, -1 };   // 1
            data[8] = new int[] { -1, 1, 1, -1, -1, 0, -1, -1 };   // 1
            data[9] = new int[] { 2, 0, 2, -1, -1, 2, -1, -1 };   // 2

            //
            // model variables
            //

            Range n = new Range(numItems).Named("Item");
            Range m = new Range(numWorkers).Named("Worker");

            var T = Variable.Array<int>(n).Named("TrueLabels");
            var S = Variable.Array(Variable.Array<bool>(m), n).Named("IsSpammer");
            var A = Variable.Array(Variable.Array<int>(m), n).Named("Answer");

            //
            // Parameters and their priors
            //

            var theta = Variable.Array<double>(m).Named("trust");
            theta[m] = Variable.Random(new Beta(2, 2)).ForEach(m);

            double[] initCounts = Enumerable.Repeat<double>(1.0, numCategories).ToArray();
            var ksi = Variable.Array<Vector>(m).Named("ksi");
            ksi[m] = Variable.Random(new Dirichlet(initCounts)).ForEach(m);

            //
            // Generative model
            //

            using (Variable.ForEach(n))
            {
                T[n] = Variable.DiscreteUniform(numCategories);
                using (Variable.ForEach(m))
                {
                    S[n][m] = Variable.Bernoulli(theta[m]);
                    using (Variable.If(A[n][m] > -1))
                    {
                        using (Variable.If(S[n][m] == false))
                        {
                            A[n][m] = T[n];
                        }
                        using (Variable.If(S[n][m] == true))
                        {
                            A[n][m] = Variable.Discrete(ksi[m]);
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

            Console.WriteLine("*** WORK ITEM LABELS ***");
            Discrete[] TMarginal = engine.Infer<Discrete[]>(T);
            for (int item = 0; item < numItems; item++)
                Console.WriteLine("\t" + TMarginal[item]);

            Console.WriteLine("\n*** IS SPAMMER ***");
            Bernoulli[][] SMarginal = engine.Infer<Bernoulli[][]>(S);
            for (int worker = 0; worker < numWorkers; worker++)
            {
                Console.WriteLine("Worker #{0}", worker);
                for (int item = 0; item < numItems; item++)
                {
                    Console.WriteLine("\t" + SMarginal[item][worker]);
                }
            }

            Console.Read();
        }
    }
}
