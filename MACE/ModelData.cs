﻿using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;

namespace MACE
{
    public class ModelData
    {
        public Discrete[] T_dist;
        public Bernoulli[][] S_dist;

        public ModelData()
        {
        }

    }
}
