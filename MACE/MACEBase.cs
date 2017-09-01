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
        protected VariableArray<Discrete> T_dist;
        protected VariableArray<int> T;

        // item-worker spam patterns
        protected VariableArray<VariableArray<Bernoulli>, Bernoulli[][]> S_dist;
        protected VariableArray<VariableArray<bool>, bool[][]> S;

        //
        // worker profile
        //

        // priors
        protected VariableArray<Bernoulli> theta_dist;
        protected VariableArray<Dirichlet> ksi_dist;

        // variables
        protected VariableArray<bool> theta;
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

            T_dist = Variable.Array<Discrete>(n).Named("T_dist");
            S_dist = Variable.Array(Variable.Array<Bernoulli>(m), n).Named("S_dist");
            theta_dist = Variable.Array<Bernoulli>(m).Named("theta_dist");
            ksi_dist = Variable.Array<Dirichlet>(m).Named("ksi_dist");

            T = Variable.Array<int>(n).Named("T");
            S = Variable.Array(Variable.Array<bool>(m), n).Named("S");
            theta = Variable.Array<bool>(m).Named("theta");
            ksi = Variable.Array<Vector>(m).Named("ksi");
        }

        public virtual void CreateModel()
        {
            if (InferenceEngine == null)
                InferenceEngine = new InferenceEngine();
        }

        public void InitializeLabels(int numItems, int numCategories)
        {
            // initialize true labels array with random label assignments to break symmetry
            Discrete[] Tinit = new Discrete[numItems];
            for (int item = 0; item < numItems; item++)
                Tinit[item] = Discrete.PointMass(Rand.Int(numCategories), numCategories);
            T.InitialiseTo(Distribution<int>.Array(Tinit));
        }

        public virtual void SetModelData(ModelData priors)
        {
            T_dist.ObservedValue = priors.T_dist;
            S_dist.ObservedValue = priors.S_dist;
            theta_dist.ObservedValue = priors.theta_dist;
            ksi_dist.ObservedValue = priors.ksi_dist;
        }
    }
}
