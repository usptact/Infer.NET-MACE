using MicrosoftResearch.Infer.Distributions;

namespace MACE
{
    public class ModelData
    {
        public Discrete[] Tprior;
        public Bernoulli[][] Sprior;

        public ModelData()
        {
        }

    }
}
