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
    public class ModelData
    {
        public Discrete[] Sprior;
        public Bernoulli[][] Tprior;

        public ModelData()
        {
        }

    }
}
