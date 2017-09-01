using System.Linq;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Maths;

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

            using (Variable.ForEach(n))
            {
                T_dist[n] = Discrete.Uniform(numCategories);
                using (Variable.ForEach(m))
                    S_dist[n][m] = Bernoulli.Uniform();
            }

            // init true label RV priors -- uniform discrete
            using (Variable.ForEach(m))
            {
                theta_dist[m] = Bernoulli.Uniform();
                ksi_dist[m] = Dirichlet.Uniform(numCategories);
				theta[m] = Variable.Random<bool, Bernoulli>(theta_dist[m]);
				ksi[m] = Variable.Random<Vector, Dirichlet>(ksi_dist[m]);
            }

            A = Variable.Array(Variable.Array<int>(m), n);
        }

        public override void CreateModel()
        {
            base.CreateModel();

            using (Variable.ForEach(n))
            {
                T[n] = Variable.Random<int, Discrete>(T_dist[n]);
                using (Variable.ForEach(m))
                {
                    S[n][m] = Variable.Random<bool, Bernoulli>(theta_dist[m]);
                    using (Variable.If(A[n][m] > -1))               // loop over observed data only
                    {
                        using (Variable.If(S[n][m] == false))
                            A[n][m] = T[n];                         // not spammer: assign true label
                        using (Variable.If(S[n][m] == true))
                            A[n][m] = Variable.Discrete(ksi[m]);    // spammer: assign label according to his profile
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
            posteriors.T_dist = InferenceEngine.Infer<Discrete[]>(T);
            posteriors.S_dist = InferenceEngine.Infer<Bernoulli[][]>(S);
            posteriors.theta_dist = InferenceEngine.Infer<Bernoulli[]>(theta);
            posteriors.ksi_dist = InferenceEngine.Infer<Dirichlet[]>(ksi);
            return posteriors;
        }
    }
}
