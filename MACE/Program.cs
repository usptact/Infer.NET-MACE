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
            const int numWorkers = 10;
            const int numItems = 50;
            const int numCategories = 3;

            int[][] data = new int[numItems][];

            Range n = new Range(numItems);
            Range m = new Range(numWorkers);

            var S = Variable.Array(Variable.Array<bool>(m), n);
            var A = Variable.Array(Variable.Array<int>(m), n);
            A.ObservedValue = data;

            Beta thetaPrior = new Beta(1, 1);
            var theta = Variable.Array<double>(m);
            theta[m] = Variable.Random(thetaPrior).ForEach(m);

            double[] initCounts = Enumerable.Repeat<double>(1.0, numCategories).ToArray();
            Dirichlet ksiPrior = new Dirichlet(initCounts);
            var ksi = Variable.Array<Vector>(m);
            ksi[m] = Variable.Random(ksiPrior).ForEach(m);

            using (Variable.ForEach(n))
            {
                var T = Variable.DiscreteUniform(numCategories);
                using (Variable.ForEach(m))
                {
                    S[n][m] = Variable.Bernoulli(1 - theta[m]);
                    using (Variable.If(S[n][m]))
                    {
                        A[n][m] = T;
                    }
                    using (Variable.IfNot(S[n][m]))
                    {
                        A[n][m] = Variable.Discrete(ksi[m]);
                    }
                }
            }

            InferenceEngine engine = new InferenceEngine();
        }
    }
}
