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

        protected VariableArray<int> T;                             // true item labels
        protected VariableArray<VariableArray<bool>, bool[][]> S;   // item-worker spam pattern

        protected VariableArray<double> theta;                      // is spammer indicator
        protected VariableArray<Vector> ksi;                        // spamming pattern per worker

        protected Range n;
        protected Range m;

        protected VariableArray<Discrete> Tprior;
        protected VariableArray<VariableArray<Bernoulli>, Bernoulli[][]> Sprior;

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

            // priors
            Tprior[m] = Variable.New<Discrete>().ForEach(m);
            Sprior[n][m] = Variable.New<Bernoulli>().ForEach(n).ForEach(m);

            // model variables (hidden RVs)
            T[m] = Variable.Random<int, Discrete>(Tprior[m]);
            S[n][m] = Variable.Random<bool, Bernoulli>(Sprior[n][m]);

            // model parameters
            theta[m] = Variable.New<double>();
            ksi[m] = Variable.New<Vector>();

            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine();
            }
        }

        public virtual void SetModelData(ModelData priors)
        {
            Tprior.ObservedValue = priors.Tprior;
            Sprior.ObservedValue = priors.Sprior;
        }
    }
}
