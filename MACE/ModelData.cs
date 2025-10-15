using Microsoft.ML.Probabilistic.Distributions;

namespace MACE
{
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
        /// Posterior distributions for true item labels (T).
        /// Each element represents the Discrete distribution over possible labels for one item.
        /// </summary>
        public Discrete[] TDist { get; set; } = Array.Empty<Discrete>();

        /// <summary>
        /// Posterior distributions for worker-item spammer indicators (S).
        /// SDist[item][worker] gives the Bernoulli distribution for whether the worker is spamming on that item.
        /// </summary>
        public Bernoulli[][] SDist { get; set; } = Array.Empty<Bernoulli[]>();

        /// <summary>
        /// Initializes a new instance of the ModelData class.
        /// </summary>
        public ModelData()
        {
        }
    }
}
