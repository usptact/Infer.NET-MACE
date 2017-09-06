using System.Linq;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;

namespace MACE
{
    public class MACETrain : MACEBase
    {
        // worker-item matrix with votes -- (partially) observed
        protected VariableArray<VariableArray<int>, int[][]> A;
        

        public MACETrain(int numWorkers, int numItems, int numCategories)
        {
            A = Variable.Array(Variable.Array<int>(m), n);

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
                    using (Variable.If(A[n][m] > -1))               // loop over observed data only
                    {
                        using (Variable.If(S[n][m] == false))
                            A[n][m] = T[n];                         // not spammer: assign true label
                        using (Variable.If(S[n][m] == true))
                            A[n][m] = Variable.Discrete(phi[m]);   // spammer: assign label according to his profile
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

            posteriors.thetaDist = InferenceEngine.Infer<Beta[]>(theta);
            posteriors.phiDist = InferenceEngine.Infer<Dirichlet[]>(phi);

            posteriors.TDist = InferenceEngine.Infer<Discrete[]>(T);
            posteriors.SDist = InferenceEngine.Infer<Bernoulli[][]>(S);

            return posteriors;
        }
    }
}
