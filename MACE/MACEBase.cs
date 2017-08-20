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

            Tprior = Variable.Array<Discrete>(n);
            Sprior[n][m] = Variable.New<Bernoulli>().ForEach(n).ForEach(m);

            T[n] = Variable.Random<int, Discrete>(Tprior[n]).ForEach(n);
            S = VariableArray.Random<bool, Bernoulli[][]>(Sprior);

            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine();
            }
        }

        public virtual void SetModelData(ModelData priors)
        {

        }
    }
}
