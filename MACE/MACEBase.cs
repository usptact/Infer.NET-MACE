using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Algorithms;

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
        protected InferenceEngine InferenceEngine { get; set; } = null!;

        /// <summary>Number of workers (columns) in the dataset.</summary>
        protected Variable<int> _numWorkers;

        /// <summary>Number of items (rows) in the dataset.</summary>
        protected Variable<int> _numItems;

        /// <summary>Number of label categories.</summary>
        protected Variable<int> _numCategories;

        /// <summary>T: the true label of each item, the main quantity being inferred.</summary>
        protected VariableArray<int> _trueLabels;

        /// <summary>
        /// Symmetry-breaking initialisation for <see cref="_trueLabels"/>, supplied as an observed
        /// value rather than baked into the compiled algorithm. See <see cref="InitializeLabels"/>.
        /// </summary>
        protected Variable<IDistribution<int[]>> _trueLabelsInit;

        /// <summary>Priors for each worker's spammer probability (theta).</summary>
        protected VariableArray<Beta> _thetaPriors;

        /// <summary>Priors for each worker's label preferences when spamming (phi).</summary>
        protected VariableArray<Dirichlet> _phiPriors;

        /// <summary>Theta: each worker's probability of spamming on any given item.</summary>
        protected VariableArray<double> _theta;

        /// <summary>Phi: the label distribution a worker draws from when spamming.</summary>
        protected VariableArray<Vector> _phi;

        /// <summary>Range over items, used as the outer index of the annotation arrays.</summary>
        protected Microsoft.ML.Probabilistic.Models.Range _itemRange;

        /// <summary>Range over workers, used to index the per-worker parameters.</summary>
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

            _trueLabels = Variable.Array<int>(_itemRange).Named("T");

            _trueLabelsInit = Variable.New<IDistribution<int[]>>().Named("TInit");
            _trueLabels.InitialiseTo(_trueLabelsInit);

            _thetaPriors = Variable.Array<Beta>(_workerRange).Named("thetaPrior");
            _phiPriors = Variable.Array<Dirichlet>(_workerRange).Named("phiPrior");

            _theta = Variable.Array<double>(_workerRange).Named("theta");
            _phi = Variable.Array<Vector>(_workerRange).Named("phi");
        }

        /// <summary>
        /// Creates the probabilistic model structure.
        /// Called once during construction by the concrete subclass after all fields are initialised.
        /// </summary>
        protected virtual void CreateModel()
        {
            // Define the prior distributions for worker parameters
            using (Variable.ForEach(_workerRange))
            {
                _theta[_workerRange] = Variable.Random<double, Beta>(_thetaPriors[_workerRange]);
                _phi[_workerRange] = Variable.Random<Vector, Dirichlet>(_phiPriors[_workerRange]);
            }

            // Initialize inference engine if not already done.
            //
            // Expectation Propagation is chosen deliberately, not inherited as a default.
            // This model cannot run under Variational Message Passing: the non-spammer branch
            // copies the true label deterministically (A = T), so under VMP each annotator
            // sends a point mass at its own observed label, and the product of those messages
            // over any two annotators who disagree is zero -- inference aborts with
            // "The model has zero probability" in ReplicateOp.DefAverageLogarithm.
            // Supporting VMP would require replacing the deterministic copy with a soft
            // confusion-matrix formulation, which is a change to the model, not the engine.
            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine(new ExpectationPropagation());
            }
        }

        /// <summary>
        /// Sets the prior distributions for the model parameters.
        /// </summary>
        /// <param name="priors">Prior distributions for worker parameters.</param>
        /// <exception cref="ArgumentNullException">Thrown when priors is null.</exception>
        protected virtual void SetModelData(ModelPriors priors)
        {
            if (priors == null)
                throw new ArgumentNullException(nameof(priors));

            _thetaPriors.ObservedValue = priors.ThetaDist;
            _phiPriors.ObservedValue = priors.PhiDist;
        }

        /// <summary>
        /// Initializes the true labels with random assignments to break symmetry.
        /// Must be called before each inference run to avoid getting stuck in a symmetric fixed point.
        /// </summary>
        /// <remarks>
        /// The initialisation is passed through the observed variable <c>TInit</c> rather than
        /// handed to <c>InitialiseTo</c> as a literal distribution. A literal is compiled into the
        /// generated algorithm as a constant, so the cached algorithm keeps the first run's values
        /// and every later run in the same process silently reuses them -- which made the seed look
        /// effective from the command line (one run per process) while doing nothing in a loop.
        /// </remarks>
        /// <param name="numItems">Number of items in the dataset.</param>
        /// <param name="numCategories">Number of possible label categories.</param>
        protected void InitializeLabels(int numItems, int numCategories)
        {
            if (numItems <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numItems), "Number of items must be positive.");
            }

            if (numCategories <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numCategories), "Number of categories must be positive.");
            }

            // Initialize true labels array with random label assignments to break symmetry
            var initialLabels = new Discrete[numItems];
            for (int item = 0; item < numItems; item++)
            {
                // Randomly assign a label to break symmetry
                int randomLabel = Rand.Int(numCategories);
                initialLabels[item] = Discrete.PointMass(randomLabel, numCategories);
            }
            
            _trueLabelsInit.ObservedValue = Distribution<int>.Array(initialLabels);
        }
    }
}
