using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;

namespace MACE
{
    /// <summary>
    /// Base class for the MACE (Multi-Annotator Competence Estimation) model.
    /// Defines the probabilistic model structure for sensor reliability estimation and threat-level inference.
    /// </summary>
    public abstract class MACEBase
    {
        /// <summary>
        /// The inference engine used for probabilistic inference.
        /// </summary>
        public InferenceEngine InferenceEngine { get; protected set; } = null!;

        // Model dimensions
        protected Variable<int> _numSensorTypes;
        protected Variable<int> _numIncidents;
        protected Variable<int> _numThreatLevels;
        
        // Data-specific model variables
        protected VariableArray<int> _threatLevels;  // T: true threat level for each incident
        protected VariableArray<VariableArray<bool>, bool[][]> _faultIndicators; // S: fault indicator per sensor per incident

        // Prior distributions for shared random variables
        protected VariableArray<Beta> _thetaPriors; // θ priors — sensor fault rate per sensor type
        protected VariableArray<Dirichlet> _phiPriors; // φ priors — sensor fault bias per sensor type

        // Shared random variables
        protected VariableArray<double> _theta; // θ — per-sensor fault rates
        protected VariableArray<Vector> _phi;   // φ — per-sensor fault-bias distributions

        // Ranges for indexing
        protected Microsoft.ML.Probabilistic.Models.Range _incidentRange;
        protected Microsoft.ML.Probabilistic.Models.Range _sensorRange;

        /// <summary>
        /// Initializes a new instance of the MACEBase class.
        /// </summary>
        protected MACEBase()
        {
            _numSensorTypes  = Variable.New<int>();
            _numIncidents    = Variable.New<int>();
            _numThreatLevels = Variable.New<int>();

            _incidentRange = new Microsoft.ML.Probabilistic.Models.Range(_numIncidents).Named("incident");
            _sensorRange   = new Microsoft.ML.Probabilistic.Models.Range(_numSensorTypes).Named("sensor");

            _threatLevels    = Variable.Array<int>(_incidentRange);
            _faultIndicators = Variable.Array(Variable.Array<bool>(_sensorRange), _incidentRange);

            _thetaPriors = Variable.Array<Beta>(_sensorRange).Named("thetaPrior");
            _phiPriors   = Variable.Array<Dirichlet>(_sensorRange).Named("phiPrior");

            _theta = Variable.Array<double>(_sensorRange).Named("theta");
            _phi   = Variable.Array<Vector>(_sensorRange).Named("phi");
        }

        /// <summary>
        /// Creates the probabilistic model structure.
        /// This method should be overridden by derived classes to define the complete model.
        /// </summary>
        public virtual void CreateModel()
        {
            // Define the prior distributions for sensor parameters (θ and φ)
            using (Variable.ForEach(_sensorRange))
            {
                _theta[_sensorRange] = Variable.Random<double, Beta>(_thetaPriors[_sensorRange]);
                _phi[_sensorRange]   = Variable.Random<Vector, Dirichlet>(_phiPriors[_sensorRange]);
            }

            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine
                {
                    // Suppress the "Iterating: 1. 2. ..." progress output that
                    // Infer.NET writes directly to Console.  We are a service;
                    // all diagnostic output goes through the structured logger.
                    ShowProgress = false,
                    ShowTimings  = false
                };

                // Keep generated C# in memory; do not write to the GeneratedSource/
                // directory (irrelevant in a container and pollutes the filesystem).
                InferenceEngine.Compiler.GenerateInMemory = true;
                InferenceEngine.Compiler.WriteSourceFiles = false;
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
        public void InitializeLabels(int numIncidents, int numThreatLevels)
            => InitializeLabels(numIncidents, numThreatLevels, null);

        /// <summary>
        /// Initializes the true labels for VMP symmetry-breaking.
        /// When <paramref name="warmStart"/> is provided its distributions are used directly,
        /// allowing the solver to start near a previous posterior and converge in fewer iterations.
        /// Falls back to random point-mass initialization when <paramref name="warmStart"/> is null.
        /// </summary>
        /// <param name="numIncidents">Number of incidents (must equal warmStart.Length when warm-starting).</param>
        /// <param name="numThreatLevels">Number of discrete threat levels.</param>
        /// <param name="warmStart">
        /// Optional prior distributions from a previous inference cycle on the same incident.
        /// Null triggers the original random initialization.
        /// </param>
        public void InitializeLabels(int numIncidents, int numThreatLevels, Discrete[]? warmStart)
        {
            if (numIncidents <= 0)
                throw new ArgumentOutOfRangeException(nameof(numIncidents), "Number of incidents must be positive.");
            if (numThreatLevels <= 0)
                throw new ArgumentOutOfRangeException(nameof(numThreatLevels), "Number of threat levels must be positive.");

            Discrete[] initialLabels;

            if (warmStart != null && warmStart.Length == numIncidents)
            {
                initialLabels = warmStart;
            }
            else
            {
                initialLabels = new Discrete[numIncidents];
                for (int incident = 0; incident < numIncidents; incident++)
                {
                    int randomLevel = Rand.Int(numThreatLevels);
                    initialLabels[incident] = Discrete.PointMass(randomLevel, numThreatLevels);
                }
            }

            _threatLevels.InitialiseTo(Distribution<int>.Array(initialLabels));
        }
    }
}
