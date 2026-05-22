using Microsoft.ML.Probabilistic.Distributions;

namespace MACE
{
    /// <summary>
    /// Prior distributions for worker parameters, passed into inference.
    /// A <see cref="ModelPosterior"/> can be used directly as priors for a subsequent inference
    /// run to support incremental/online learning.
    /// </summary>
    public record ModelPriors(
        /// <summary>Beta prior distributions for each worker's spammer probability (theta).</summary>
        Beta[] ThetaDist,
        /// <summary>Dirichlet prior distributions for each worker's label preferences when spamming (phi).</summary>
        Dirichlet[] PhiDist
    );

    /// <summary>
    /// Posterior distributions returned by inference, containing both the worker parameter
    /// posteriors and the inferred item-level distributions.
    /// </summary>
    public record ModelPosterior(
        /// <summary>Posterior Beta distributions for each worker's spammer probability (theta).</summary>
        Beta[] ThetaDist,
        /// <summary>Posterior Dirichlet distributions for each worker's label preferences when spamming (phi).</summary>
        Dirichlet[] PhiDist,
        /// <summary>Posterior Discrete distributions over the true label for each item (T).</summary>
        Discrete[] TDist,
        /// <summary>Posterior Bernoulli distributions for whether each worker is spamming on each item (S[item][worker]).</summary>
        Bernoulli[][] SDist
    ) : ModelPriors(ThetaDist, PhiDist);
}
