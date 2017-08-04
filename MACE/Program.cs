using System;
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
            const int numWorkers = 5;
            const int numItems = 10;
            const int numCategories = 3;

            int[][] data = new int[numItems][];

            data[0] = new int[] { 0, 1, 1, 2, 2 };
            data[1] = new int[] { 1, 1, 0, 1, 2 };
            data[2] = new int[] { 0, 1, 1, 2, 0 };
            data[3] = new int[] { 2, 2, 0, 2, 1 };
            data[4] = new int[] { 0, 1, 1, 1, 2 };
            data[5] = new int[] { 0, 0, 1, 2, 2 };
            data[6] = new int[] { 1, 1, 2, 2, 1 };
            data[7] = new int[] { 0, 0, 1, 2, 2 };
            data[8] = new int[] { 1, 1, 1, 0, 2 };
            data[9] = new int[] { 0, 1, 1, 2, 2 };

            //
            // model variables
            //

            Range n = new Range(numItems);
            Range m = new Range(numWorkers);

            var S = Variable.Array(Variable.Array<bool>(m), n);
            var A = Variable.Array(Variable.Array<int>(m), n);
            A.ObservedValue = data;

            //
            // Parameters and their priors
            //

            Beta thetaPrior = new Beta(1, 1);
            var theta = Variable.Array<double>(m);
            theta[m] = Variable.Random(thetaPrior).ForEach(m);

            double[] initCounts = Enumerable.Repeat<double>(1.0, numCategories).ToArray();
            Dirichlet ksiPrior = new Dirichlet(initCounts);
            var ksi = Variable.Array<Vector>(m);
            ksi[m] = Variable.Random(ksiPrior).ForEach(m);

            //
            // Generative model
            //

            using (Variable.ForEach(n))
            {
                var T = Variable.DiscreteUniform(numCategories);
                using (Variable.ForEach(m))
                {
                    S[n][m] = Variable.Bernoulli(theta[m]);
                    using (Variable.IfNot(S[n][m]))
                    {
                        A[n][m] = T;
                    }
                    using (Variable.If(S[n][m]))
                    {
                        A[n][m] = Variable.Discrete(ksi[m]);
                    }
                }
            }

            //
            // Inference
            //

            InferenceEngine engine = new InferenceEngine(new VariationalMessagePassing());

            Bernoulli[][] SMarginal = engine.Infer<Bernoulli[][]>(S);
        }
    }
}
