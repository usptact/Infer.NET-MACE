using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace MACE
{
    /// <summary>
    /// Implements the MACE (Multi-Annotator Competence Estimation) model for crowdsourcing annotation quality estimation.
    /// This class extends MACEBase to define the complete probabilistic model including the observation model.
    /// </summary>
    public class MACETrain : MACEBase
    {
        /// <summary>
        /// Worker-item annotation matrix containing the observed votes (partially observed).
        /// A[item][worker] contains the annotation for that item-worker pair, or -1 for missing annotations.
        /// </summary>
        protected VariableArray<VariableArray<int>, int[][]> _annotations;

        /// <summary>
        /// Convenience constructor for the online inference pod.
        /// <c>numItems</c> is always 1 in the online setting (one active incident per request).
        /// </summary>
        /// <param name="numSensorTypes">Number of sensor types (= workers in MACE terminology).</param>
        /// <param name="numCategories">Number of threat-level categories (5 in ThreatSense).</param>
        public MACETrain(int numSensorTypes, int numCategories)
            : this(numSensorTypes, numItems: 1, numCategories) { }

        /// <summary>
        /// Initializes a new instance of the MACETrain class.
        /// </summary>
        /// <param name="numWorkers">Number of workers in the dataset.</param>
        /// <param name="numItems">Number of items to be annotated.</param>
        /// <param name="numCategories">Number of possible label categories.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is non-positive.</exception>
        public MACETrain(int numWorkers, int numItems, int numCategories)
        {
            if (numWorkers <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numWorkers), "Number of workers must be positive.");
            }

            if (numItems <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numItems), "Number of items must be positive.");
            }

            if (numCategories <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numCategories), "Number of categories must be positive.");
            }

            _annotations = Variable.Array(Variable.Array<int>(_workerRange), _itemRange);

            _numWorkers.ObservedValue = numWorkers;
            _numItems.ObservedValue = numItems;
            _numCategories.ObservedValue = numCategories;
        }

        /// <summary>
        /// Creates the complete MACE probabilistic model including the observation model.
        /// </summary>
        public override void CreateModel()
        {
            base.CreateModel();

            using (Variable.ForEach(_itemRange))
            {
                // True label for this item (uniform prior over categories)
                _trueLabels[_itemRange] = Variable.DiscreteUniform(_numCategories);
                
                using (Variable.ForEach(_workerRange))
                {
                    // Spammer indicator for this worker-item pair
                    _spammerIndicators[_itemRange][_workerRange] = Variable.Bernoulli(_theta[_workerRange]);
                    
                    // Only process observed annotations (skip missing data marked with -1)
                    using (Variable.If(_annotations[_itemRange][_workerRange] > -1))
                    {
                        using (Variable.If(_spammerIndicators[_itemRange][_workerRange] == false))
                        {
                            // Not a spammer: assign the true label
                            _annotations[_itemRange][_workerRange] = _trueLabels[_itemRange];
                        }
                        
                        using (Variable.If(_spammerIndicators[_itemRange][_workerRange] == true))
                        {
                            // Spammer: assign label according to their preference distribution
                            _annotations[_itemRange][_workerRange] = Variable.Discrete(_phi[_workerRange]);
                        }
                    }
                }
            }

            // Prevent the inference engine from trying to infer the observed annotations
            // The annotations matrix can contain -1 values which are out of the domain
            _annotations.AddAttribute(new DoNotInfer());
        }

        /// <summary>
        /// Performs probabilistic inference to estimate the posterior distributions of all model parameters.
        /// </summary>
        /// <param name="data">The annotation data matrix where data[item][worker] gives the annotation or -1 for missing.</param>
        /// <returns>ModelData containing the posterior distributions for all model parameters.</returns>
        /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
        /// <exception cref="ArgumentException">Thrown when data dimensions don't match the expected dimensions.</exception>
        public ModelData InferModelData(int[][] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            // Validate data dimensions
            if (data.Length != _numItems.ObservedValue)
            {
                throw new ArgumentException(
                    $"Data has {data.Length} items but model expects {_numItems.ObservedValue} items.", 
                    nameof(data));
            }

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == null || data[i].Length != _numWorkers.ObservedValue)
                {
                    throw new ArgumentException(
                        $"Data row {i} has {data[i]?.Length ?? 0} workers but model expects {_numWorkers.ObservedValue} workers.", 
                        nameof(data));
                }
            }

            var posteriors = new ModelData();

            // Set the observed annotation data
            _annotations.ObservedValue = data;

            // Perform inference to get posterior distributions
            posteriors.ThetaDist = InferenceEngine.Infer<Beta[]>(_theta);
            posteriors.PhiDist = InferenceEngine.Infer<Dirichlet[]>(_phi);
            posteriors.TDist = InferenceEngine.Infer<Discrete[]>(_trueLabels);
            posteriors.SDist = InferenceEngine.Infer<Bernoulli[][]>(_spammerIndicators);

            return posteriors;
        }

        /// <summary>
        /// Online single-incident inference.
        /// Accepts a flat annotation vector (one entry per sensor type, -1 = absent),
        /// injects the supplied priors, optionally warm-starts from a previous posterior,
        /// and returns a typed result that avoids batch-array indexing by the caller.
        /// </summary>
        /// <param name="annotations">
        /// Flat array of length <c>numSensorTypes</c>. -1 marks absent sensors.
        /// </param>
        /// <param name="priors">Current theta and phi priors from the Belief Store.</param>
        /// <param name="warmStart">
        /// Optional: T[0] posterior from a previous call on the same incident.
        /// Improves convergence speed on subsequent calls.
        /// </param>
        public OnlineInferenceResult InferOnline(
            int[] annotations,
            ModelData priors,
            Discrete? warmStart = null)
        {
            if (annotations is null)
                throw new ArgumentNullException(nameof(annotations));
            if (annotations.Length != _numWorkers.ObservedValue)
                throw new ArgumentException(
                    $"annotations length {annotations.Length} != numSensorTypes {_numWorkers.ObservedValue}");
            if (priors is null)
                throw new ArgumentNullException(nameof(priors));

            SetModelData(priors);

            // Discrete is a reference type — check for null directly.
            Discrete[]? warmStartArray = warmStart is not null ? [warmStart] : null;
            InitializeLabels(1, _numCategories.ObservedValue, warmStartArray);

            // Set observed data and run VMP.
            // Call Infer<> only for the two variables we need (TDist and SDist).
            // Skipping ThetaDist and PhiDist avoids allocating those arrays; VMP
            // still runs once and produces valid marginals for all variables.
            _annotations.ObservedValue = new int[1][] { annotations };
            var tDist = InferenceEngine.Infer<Discrete[]>(_trueLabels)[0];
            var sDist = InferenceEngine.Infer<Bernoulli[][]>(_spammerIndicators)[0];
            var probs  = tDist.GetProbs();

            int    threatLevel = 0;
            double confidence  = 0.0;
            for (int i = 0; i < probs.Count; i++)
            {
                if (probs[i] > confidence)
                {
                    confidence  = probs[i];
                    threatLevel = i;
                }
            }

            double entropy = 0.0;
            for (int i = 0; i < probs.Count; i++)
            {
                double p = probs[i];
                if (p > 0.0)
                    entropy -= p * Math.Log(p);
            }

            return new OnlineInferenceResult(tDist, sDist, threatLevel, confidence, entropy);
        }
    }
}
