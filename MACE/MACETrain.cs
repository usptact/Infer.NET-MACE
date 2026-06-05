using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace MACE
{
    /// <summary>
    /// Implements the MACE (Multi-Annotator Competence Estimation) model for sensor reliability estimation
    /// and threat-level inference. Extends MACEBase with the observation model (sensor readings).
    /// </summary>
    public class MACETrain : MACEBase
    {
        /// <summary>
        /// Sensor reading matrix (partially observed).
        /// A[incident][sensor] contains the sensor's discretised reading (0–4), or -1 for absent sensors.
        /// </summary>
        protected VariableArray<VariableArray<int>, int[][]> _sensorReadings;

        /// <summary>
        /// Convenience constructor for the online inference pod.
        /// <c>numIncidents</c> is fixed to 1: the pod processes one active incident per request.
        /// </summary>
        /// <param name="numSensorTypes">Number of sensor types.</param>
        /// <param name="numThreatLevels">Number of discrete threat levels (5 in ThreatSense).</param>
        public MACETrain(int numSensorTypes, int numThreatLevels)
            : this(numSensorTypes, numIncidents: 1, numThreatLevels) { }

        /// <summary>
        /// Full constructor; supports both online (numIncidents=1) and batch (numIncidents&gt;1) inference.
        /// </summary>
        /// <param name="numSensorTypes">Number of sensor types.</param>
        /// <param name="numIncidents">Number of incidents in the reading matrix. Use 1 for online inference.</param>
        /// <param name="numThreatLevels">Number of discrete threat levels.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is non-positive.</exception>
        public MACETrain(int numSensorTypes, int numIncidents, int numThreatLevels)
        {
            if (numSensorTypes <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numSensorTypes), "Number of sensor types must be positive.");
            }

            if (numIncidents <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numIncidents), "Number of incidents must be positive.");
            }

            if (numThreatLevels <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numThreatLevels), "Number of threat levels must be positive.");
            }

            _sensorReadings = Variable.Array(Variable.Array<int>(_sensorRange), _incidentRange);

            _numSensorTypes.ObservedValue  = numSensorTypes;
            _numIncidents.ObservedValue    = numIncidents;
            _numThreatLevels.ObservedValue = numThreatLevels;
        }

        /// <summary>
        /// Creates the complete MACE probabilistic model including the observation model.
        /// </summary>
        public override void CreateModel()
        {
            base.CreateModel();

            using (Variable.ForEach(_incidentRange))
            {
                // True threat level for this incident (uniform prior over threat levels)
                _threatLevels[_incidentRange] = Variable.DiscreteUniform(_numThreatLevels);

                using (Variable.ForEach(_sensorRange))
                {
                    // Fault indicator for this sensor-incident pair
                    _faultIndicators[_incidentRange][_sensorRange] = Variable.Bernoulli(_theta[_sensorRange]);

                    // Only process observed readings (skip absent sensors marked with -1)
                    using (Variable.If(_sensorReadings[_incidentRange][_sensorRange] > -1))
                    {
                        using (Variable.If(_faultIndicators[_incidentRange][_sensorRange] == false))
                        {
                            // Reliable sensor: reports the true threat level
                            _sensorReadings[_incidentRange][_sensorRange] = _threatLevels[_incidentRange];
                        }

                        using (Variable.If(_faultIndicators[_incidentRange][_sensorRange] == true))
                        {
                            // Faulty sensor: reports according to its fault-bias distribution
                            _sensorReadings[_incidentRange][_sensorRange] = Variable.Discrete(_phi[_sensorRange]);
                        }
                    }
                }
            }

            // Prevent the inference engine from trying to infer the observed sensor readings.
            // The readings matrix can contain -1 values which are out of the domain.
            _sensorReadings.AddAttribute(new DoNotInfer());
        }

        /// <summary>
        /// Performs probabilistic inference to estimate the posterior distributions of all model parameters.
        /// </summary>
        /// <param name="data">Sensor reading matrix: data[incident][sensor] gives the reading (0–4) or -1 for absent sensors.</param>
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
            if (data.Length != _numIncidents.ObservedValue)
            {
                throw new ArgumentException(
                    $"Data has {data.Length} incidents but model expects {_numIncidents.ObservedValue} incidents.",
                    nameof(data));
            }

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == null || data[i].Length != _numSensorTypes.ObservedValue)
                {
                    throw new ArgumentException(
                        $"Data row {i} has {data[i]?.Length ?? 0} sensor types but model expects {_numSensorTypes.ObservedValue} sensor types.",
                        nameof(data));
                }
            }

            var posteriors = new ModelData();

            // Set the observed sensor readings
            _sensorReadings.ObservedValue = data;

            // Perform inference to get posterior distributions
            posteriors.ThetaDist   = InferenceEngine.Infer<Beta[]>(_theta);
            posteriors.PhiDist     = InferenceEngine.Infer<Dirichlet[]>(_phi);
            posteriors.ThreatDist  = InferenceEngine.Infer<Discrete[]>(_threatLevels);
            posteriors.FaultDist   = InferenceEngine.Infer<Bernoulli[][]>(_faultIndicators);

            return posteriors;
        }

        /// <summary>
        /// Online single-incident inference.
        /// Accepts a flat sensor reading vector (one entry per sensor type, -1 = absent),
        /// injects the supplied priors, optionally warm-starts from a previous posterior,
        /// and returns a typed result that avoids batch-array indexing by the caller.
        /// </summary>
        /// <param name="annotations">
        /// Flat sensor reading array of length <c>numSensorTypes</c>. -1 marks absent sensors.
        /// Named <c>annotations</c> to match the proto field; semantically these are sensor readings.
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
            if (annotations.Length != _numSensorTypes.ObservedValue)
                throw new ArgumentException(
                    $"annotations length {annotations.Length} != numSensorTypes {_numSensorTypes.ObservedValue}");
            if (priors is null)
                throw new ArgumentNullException(nameof(priors));

            SetModelData(priors);

            // Discrete is a reference type — check for null directly.
            Discrete[]? warmStartArray = warmStart is not null ? [warmStart] : null;
            InitializeLabels(1, _numThreatLevels.ObservedValue, warmStartArray);

            // Set observed sensor readings and run VMP.
            // Call Infer<> only for the two variables we need (ThreatDist and FaultDist).
            // Skipping ThetaDist and PhiDist avoids allocating those arrays; VMP
            // still runs once and produces valid marginals for all variables.
            _sensorReadings.ObservedValue = new int[1][] { annotations };
            var threatDist = InferenceEngine.Infer<Discrete[]>(_threatLevels)[0];
            var faultDist  = InferenceEngine.Infer<Bernoulli[][]>(_faultIndicators)[0];
            var probs      = threatDist.GetProbs();

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

            return new OnlineInferenceResult(threatDist, faultDist, threatLevel, confidence, entropy);
        }
    }
}
