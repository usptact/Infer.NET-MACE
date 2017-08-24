using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace MACE
{
    public class MACEBase
    {
        public InferenceEngine InferenceEngine;

        protected Variable<int> numWorkers;
        protected Variable<int> numItems;
        protected Variable<int> numCategories;

        protected VariableArray<int> T;
        protected VariableArray<VariableArray<bool>, bool[][]> S;

        protected Range n;
        protected Range m;

        protected VariableArray<Discrete> Sprior;
        protected VariableArray<Bernoulli[]> Tprior;

        public MACEBase()
        {
        }

        public virtual void CreateModel()
        {
            numWorkers = Variable.New<int>();
            numItems = Variable.New<int>();
            numCategories = Variable.New<int>();

            n = new Range(numItems);
            m = new Range(numWorkers);

            var T = Variable.Array<int>(n);
            var S = Variable.Array(Variable.Array<bool>(m), n);

            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine();
            }
        }

        public virtual void SetModelData(ModelData priors)
        {
            Sprior.ObservedValue = priors.Sprior; // "worker trust"
            Tprior.ObservedValue = priors.Tprior;     // "spammer's preferences"
        }
    }
}
