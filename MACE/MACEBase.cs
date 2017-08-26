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

        // true item labels
        protected VariableArray<Discrete> Tprior;
        protected VariableArray<int> T;

        // item-worker spam patterns
        protected VariableArray<VariableArray<Bernoulli>, Bernoulli[][]> Sprior;
        protected VariableArray<VariableArray<bool>, bool[][]> S;

        protected VariableArray<double> theta;                      // is spammer indicator
        protected VariableArray<Vector> ksi;                        // spamming pattern per worker

        protected Range n;
        protected Range m;


        public MACEBase()
        {
            numWorkers = Variable.New<int>();
            numItems = Variable.New<int>();
            numCategories = Variable.New<int>();

            n = new Range(numItems);
            m = new Range(numWorkers);

            Tprior = Variable.Array<Discrete>(m);
            T = Variable.Array<int>(n);

            Sprior = Variable.Array(Variable.Array<Bernoulli>(m), n);
            S = Variable.Array(Variable.Array<bool>(m), n);

            theta = Variable.Array<double>(m);
            ksi = Variable.Array<Vector>(m);
        }

        public virtual void CreateModel()
        {
            // true label
            using (Variable.ForEach(m))
            {
                Tprior[m] = Variable.New<Discrete>();
                T[m] = Variable.Random<int, Discrete>(Tprior[m]);
            }

            // worker-item spamming patterns
            using (Variable.ForEach(n))
            {
                using (Variable.ForEach(m))
                {
                    Sprior[n][m] = Variable.New<Bernoulli>();
                    S[n][m] = Variable.Random<bool, Bernoulli>(Sprior[n][m]);
                }
            }

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
