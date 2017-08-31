using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;

namespace MACE
{
    public class ModelData
    {
        public VariableArray<Discrete> T_dist;
        public VariableArray<Bernoulli> theta_dist;
        public VariableArray<Dirichlet> ksi_dist;

        public ModelData()
        {
        }

    }
}
