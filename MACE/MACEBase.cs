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

        protected VariableArray<Discrete> Tprior;
        protected VariableArray<VariableArray<Bernoulli>, Bernoulli[][]> Sprior;

        protected VariableArray<int> T;
        protected VariableArray<VariableArray<bool>, bool[][]> S;
        protected VariableArray<VariableArray<int>, int[][]> A;      // move to MACETrain?

        protected Range n;
        protected Range m;

        protected VariableArray<Beta> theta;
        protected VariableArray<Dirichlet> ksi;

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

            // no -- set priors instead
            Tprior = Variable.Array<Discrete>(n);
            Sprior[n][m] = Variable.New<Bernoulli>().ForEach(n).ForEach(m);

            // no -- set priors of T and S instead
            T[n] = Variable.Random<int, Discrete>(Tprior[n]);
            S[n][m] = Variable.Random<bool, Bernoulli>(Sprior[n][m]);



            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine();
            }
        }

        public virtual void SetModelData(ModelData priors)
        {
            //T.ObservedValue = priors.Tprior;    // true labels
            //S.ObservedValue = priors.Sprior;    // "is spammer"
            theta.ObservedValue = priors.theta; // "worker trust"
            ksi.ObservedValue = priors.ksi;     // "spammer's preferences"
        }
    }
}
