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

            // init true label RV priors -- uniform discrete
            for (int i = 0; i < numItems; i++)
                T_dist[i] = new Discrete(Enumerable.Repeat<double>(1.0 / numCategories, numCategories).ToArray());

            // init worker RV priors
            for (int j = 0; j < numWorkers; j++)
            {
                theta_dist[j] = new Bernoulli(0.5);
                ksi_dist[j] = new Dirichlet(Enumerable.Repeat<double>(1.0, numCategories).ToArray());
            }

            // attaching priors to respective RVs
            using (Variable.ForEach(m))
            {
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
                    //S[n][m] = Variable.Random<bool, Bernoulli>(S_dist[n][m]);
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
            posteriors.T_dist.ObservedValue = InferenceEngine.Infer<Discrete[]>(T);
            posteriors.theta_dist.ObservedValue = InferenceEngine.Infer<Bernoulli[]>(theta);
            posteriors.ksi_dist.ObservedValue = InferenceEngine.Infer<Dirichlet[]>(ksi);
            return posteriors;
        }
    }
}
