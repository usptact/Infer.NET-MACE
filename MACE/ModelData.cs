using Microsoft.ML.Probabilistic.Distributions;

namespace MACE
{
    /// <summary>
    /// Typed result of a single-incident online inference call.
    /// Unwraps the batch arrays so callers never index <c>[0]</c> manually.
    /// </summary>
    public record OnlineInferenceResult(
        Discrete    ThreatDist,  // posterior over threat levels (length = NumThreatLevels)
        Bernoulli[] FaultDist,   // per-sensor fault indicator posteriors
        int         ThreatLevel, // argmax(ThreatDist.GetProbs())
        double      Confidence,  // max probability
        double      Entropy      // Shannon entropy (uncertainty measure)
    );


    /// <summary>
    /// Contains the distributions for the MACE model parameters.
    /// This class holds both prior and posterior distributions for the model variables.
    /// </summary>
    public class ModelData
    {
        /// <summary>
        /// Prior/posterior distributions for worker spammer probabilities (theta).
        /// Each element represents the Beta distribution for one worker's spammer probability.
        /// </summary>
        public Beta[] ThetaDist { get; set; } = Array.Empty<Beta>();

        /// <summary>
        /// Prior/posterior distributions for worker label preferences (phi).
        /// Each element represents the Dirichlet distribution for one worker's label preferences when spamming.
        /// </summary>
        public Dirichlet[] PhiDist { get; set; } = Array.Empty<Dirichlet>();

        /// <summary>
        /// Posterior distributions for true threat levels (T).
        /// Each element represents the Discrete distribution over threat levels for one incident.
        /// </summary>
        public Discrete[] ThreatDist { get; set; } = Array.Empty<Discrete>();

        /// <summary>
        /// Posterior fault-indicator distributions (S).
        /// FaultDist[incident][sensor] gives the Bernoulli distribution for whether
        /// the sensor produced a faulty reading for that incident.
        /// </summary>
        public Bernoulli[][] FaultDist { get; set; } = Array.Empty<Bernoulli[]>();

        /// <summary>
        /// Initializes a new instance of the ModelData class.
        /// </summary>
        public ModelData()
        {
        }
    }
}
