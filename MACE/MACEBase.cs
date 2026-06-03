using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;

namespace MACE
{
    /// <summary>
    /// Base class for the MACE (Multi-Annotator Competence Estimation) model.
    /// This class defines the probabilistic model structure for crowdsourcing annotation quality estimation.
    /// </summary>
    public abstract class MACEBase
    {
        /// <summary>
        /// The inference engine used for probabilistic inference.
        /// </summary>
        public InferenceEngine InferenceEngine { get; protected set; } = null!;

        // Model dimensions
        protected Variable<int> _numWorkers;
        protected Variable<int> _numItems;
        protected Variable<int> _numCategories;
        
        // Data-specific model variables
        protected VariableArray<int> _trueLabels; // T: true labels for each item
        protected VariableArray<VariableArray<bool>, bool[][]> _spammerIndicators; // S: whether each worker is spamming on each item

        // Prior distributions for shared random variables
        protected VariableArray<Beta> _thetaPriors; // Priors for worker spammer probabilities
        protected VariableArray<Dirichlet> _phiPriors; // Priors for worker label preferences when spamming

        // Shared random variables
        protected VariableArray<double> _theta; // Worker spammer probabilities
        protected VariableArray<Vector> _phi; // Worker label preferences when spamming

        // Ranges for indexing
        protected Microsoft.ML.Probabilistic.Models.Range _itemRange;
        protected Microsoft.ML.Probabilistic.Models.Range _workerRange;

        /// <summary>
        /// Initializes a new instance of the MACEBase class.
        /// </summary>
        protected MACEBase()
        {
            _numWorkers = Variable.New<int>();
            _numItems = Variable.New<int>();
            _numCategories = Variable.New<int>();

            _itemRange = new Microsoft.ML.Probabilistic.Models.Range(_numItems).Named("item");
            _workerRange = new Microsoft.ML.Probabilistic.Models.Range(_numWorkers).Named("worker");

            _trueLabels = Variable.Array<int>(_itemRange);
            _spammerIndicators = Variable.Array(Variable.Array<bool>(_workerRange), _itemRange);

            _thetaPriors = Variable.Array<Beta>(_workerRange).Named("thetaPrior");
            _phiPriors = Variable.Array<Dirichlet>(_workerRange).Named("phiPrior");

            _theta = Variable.Array<double>(_workerRange).Named("theta");
            _phi = Variable.Array<Vector>(_workerRange).Named("phi");
        }

        /// <summary>
        /// Creates the probabilistic model structure.
        /// This method should be overridden by derived classes to define the complete model.
        /// </summary>
        public virtual void CreateModel()
        {
            // Define the prior distributions for worker parameters
            using (Variable.ForEach(_workerRange))
            {
                _theta[_workerRange] = Variable.Random<double, Beta>(_thetaPriors[_workerRange]);
                _phi[_workerRange] = Variable.Random<Vector, Dirichlet>(_phiPriors[_workerRange]);
            }

            // Initialize inference engine if not already done
            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine();
            }
        }

        /// <summary>
        /// Sets the prior distributions for the model parameters.
        /// </summary>
        /// <param name="modelData">ModelData containing the prior distributions.</param>
        /// <exception cref="ArgumentNullException">Thrown when modelData is null.</exception>
        public virtual void SetModelData(ModelData modelData)
        {
            if (modelData == null)
            {
                throw new ArgumentNullException(nameof(modelData));
            }

            _thetaPriors.ObservedValue = modelData.ThetaDist;
            _phiPriors.ObservedValue = modelData.PhiDist;
        }

        /// <summary>
        /// Initializes the true labels with random assignments to break symmetry.
        /// Delegates to the warm-start overload with <c>warmStart = null</c>.
        /// </summary>
        public void InitializeLabels(int numItems, int numCategories)
            => InitializeLabels(numItems, numCategories, null);

        /// <summary>
        /// Initializes the true labels for VMP symmetry-breaking.
        /// When <paramref name="warmStart"/> is provided its distributions are used directly,
        /// allowing the solver to start near a previous posterior and converge in fewer iterations.
        /// Falls back to random point-mass initialization when <paramref name="warmStart"/> is null.
        /// </summary>
        /// <param name="numItems">Number of items (must equal warmStart.Length when warm-starting).</param>
        /// <param name="numCategories">Number of possible label categories.</param>
        /// <param name="warmStart">
        /// Optional prior distributions from a previous inference cycle on the same incident.
        /// Null triggers the original random initialization.
        /// </param>
        public void InitializeLabels(int numItems, int numCategories, Discrete[]? warmStart)
        {
            if (numItems <= 0)
                throw new ArgumentOutOfRangeException(nameof(numItems), "Number of items must be positive.");
            if (numCategories <= 0)
                throw new ArgumentOutOfRangeException(nameof(numCategories), "Number of categories must be positive.");

            Discrete[] initialLabels;

            if (warmStart != null && warmStart.Length == numItems)
            {
                initialLabels = warmStart;
            }
            else
            {
                initialLabels = new Discrete[numItems];
                for (int item = 0; item < numItems; item++)
                {
                    int randomLabel = Rand.Int(numCategories);
                    initialLabels[item] = Discrete.PointMass(randomLabel, numCategories);
                }
            }

            _trueLabels.InitialiseTo(Distribution<int>.Array(initialLabels));
        }
    }
}
