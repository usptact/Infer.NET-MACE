using System;
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

            int[][] data = new int[numWorkers][];

            Range n = new Range(numItems);
            Range m = new Range(numWorkers);

            var S = Variable.Array(Variable.Array<bool>(m), n);
            var A = Variable.Array(Variable.Array<int>(m), n);

            using (Variable.ForEach(n))
            {
                var T = Variable.DiscreteUniform(numCategories);
                using (Variable.ForEach(m))
                {
                    S[n][m] = Variable.Bernoulli(1 - theta[j]);
                    using (Variable.If(S[n][m]))
                    {
                        A[n][m] = T;
                    }
                    using (Variable.IfNot(S[n][m]))
                    {
                        A[n][m] = Variable.Multinomial()
                    }
                }
            }
        }
    }
}
