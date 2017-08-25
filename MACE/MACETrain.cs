using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace MACE
{
    public class MACETrain : MACEBase
    {
        protected VariableArray<VariableArray<int>, int[][]> A;

        public MACETrain(int numWorkers, int numItems, int numCategories)
        {
            this.numWorkers.ObservedValue = numWorkers;
            this.numItems.ObservedValue = numItems;
            this.numCategories.ObservedValue = numCategories;
        }

        public override void CreateModel()
        {
            base.CreateModel();

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

            A.AddAttribute(new DoNotInfer());
        }

        public ModelData InferModelData(int[][] data)
        {
            ModelData posteriors = new ModelData();
            A.ObservedValue = data;
            posteriors.Tprior = InferenceEngine.Infer<Discrete[]>(T);
            posteriors.Sprior = InferenceEngine.Infer<Bernoulli[][]>(S);
            return posteriors;
        }
    }
}
