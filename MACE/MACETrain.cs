using System.Linq;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
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

            // set uniform priors
            using (Variable.ForEach(n))
            {
                Tprior[n] = new Discrete(Enumerable.Repeat<double>(1.0 / numCategories, numCategories).ToArray());
                using (Variable.ForEach(m))
                {
                    Sprior[n][m] = new Bernoulli(0.5);
                }
            }

            using (Variable.ForEach(m))
            {
                theta[m] = Variable.Random(new Beta(1, 1));
                ksi[m] = Variable.Random(new Dirichlet(Enumerable.Repeat<double>(1.0, numCategories).ToArray()));
            }

            A = Variable.Array(Variable.Array<int>(m), n);
        }

        public override void CreateModel()
        {
            base.CreateModel();

            using (Variable.ForEach(n))
            {
                //T[n] = Variable.DiscreteUniform(numCategories);
                T[n] = Variable.Random<int, Discrete>(Tprior[n]);
                using (Variable.ForEach(m))
                {
                    //S[n][m] = Variable.Bernoulli(theta[m]);
                    S[n][m] = Variable.Random<bool, Bernoulli>(Sprior[n][m]);
                    using (Variable.If(A[n][m] > -1))
                    {
                        using (Variable.If(S[n][m] == false))
                        {
                            A[n][m] = T[n];                         // not spammer: assign true label
                        }
                        using (Variable.If(S[n][m] == true))
                        {
                            A[n][m] = Variable.Discrete(ksi[m]);    // spammer: assign label according to his profile
                        }
                    }
                }
            }

            // prevent engine from trying to infer "A"
            // "A" can contain negative values that are out of domain
            A.AddAttribute(new DoNotInfer());
        }

        public ModelData InferModelData(int[][] data)
        {
            // !!! data dimensions must match numWorkers x numItems every call!!!
            ModelData posteriors = new ModelData();
            A.ObservedValue = data;
            posteriors.Tprior = InferenceEngine.Infer<Discrete[]>(T);
            posteriors.Sprior = InferenceEngine.Infer<Bernoulli[][]>(S);
            return posteriors;
        }
    }
}
