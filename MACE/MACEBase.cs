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
        
        // model variables
        protected VariableArray<int> T;
        protected VariableArray<VariableArray<bool>, bool[][]> S;

        protected VariableArray<double> theta;
        protected VariableArray<Vector> phi;

        protected Range n;
        protected Range m;

        public MACEBase()
        {
            numWorkers = Variable.New<int>();
            numItems = Variable.New<int>();
            numCategories = Variable.New<int>();

            n = new Range(numItems).Named("item");
            m = new Range(numWorkers).Named("worker");

            T = Variable.Array<int>(n);
            S = Variable.Array(Variable.Array<bool>(m), n);

            theta = Variable.Array<double>(m);
            phi = Variable.Array<Vector>(m);
        }

        public virtual void CreateModel()
        {
            theta[m] = Variable.Beta(1, 1).ForEach(m);
            phi[m] = Variable.DirichletUniform(numCategories).ForEach(m);

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
    }
}
