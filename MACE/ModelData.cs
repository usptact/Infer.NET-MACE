using Microsoft.ML.Probabilistic.Distributions;

namespace MACE
{
    public class ModelData
    {
        // shared model parameters
        public Beta[] thetaDist;
        public Dirichlet[] phiDist;

        // data-specific parameters
        public Discrete[] TDist;
        public Bernoulli[][] SDist;

        public ModelData()
        {
        }

    }
}
