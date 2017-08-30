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

        // worker profile
        protected VariableArray<double> theta;                      // is spammer indicator
        protected VariableArray<Vector> ksi;                        // spamming pattern per worker

        protected Range n;
        protected Range m;


        public MACEBase()
        {
            numWorkers = Variable.New<int>();
            numItems = Variable.New<int>();
            numCategories = Variable.New<int>();

            n = new Range(numItems).Named("item");
            m = new Range(numWorkers).Named("worker");

            Tprior = Variable.Array<Discrete>(n).Named("T_prior");
            T = Variable.Array<int>(n).Named("T");

            Sprior = Variable.Array(Variable.Array<Bernoulli>(m), n).Named("S_prior");
            S = Variable.Array(Variable.Array<bool>(m), n).Named("S");

            theta = Variable.Array<double>(m).Named("theta");
            ksi = Variable.Array<Vector>(m).Named("ksi");
        }

        public virtual void CreateModel()
        {
            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine();
            }
        }

        public void InitializeLabels(int numItems, int numCategories)
        {
            Discrete[] Tinit = new Discrete[numItems];
            for (int item = 0; item < numItems; item++)
                Tinit[item] = Discrete.PointMass(Rand.Int(numCategories), numCategories);
            T.InitialiseTo(Distribution<int>.Array(Tinit));
        }

        public virtual void SetModelData(ModelData priors)
        {
            Tprior.ObservedValue = priors.Tprior;
            Sprior.ObservedValue = priors.Sprior;
        }
    }
}
