using Microsoft.ML.Probabilistic.Distributions;

namespace MACE
{
    /// <summary>
    /// Prior distributions for worker parameters, passed into inference.
    /// A <see cref="ModelPosterior"/> can be used directly as priors for a subsequent inference
    /// run to support incremental/online learning, and <see cref="ModelPriorsIo"/> persists them
    /// between processes.
    /// </summary>
    /// <param name="ThetaDist">Beta prior distributions for each worker's spammer probability (theta).</param>
    /// <param name="PhiDist">Dirichlet prior distributions for each worker's label preferences when spamming (phi).</param>
    public record ModelPriors(
        Beta[] ThetaDist,
        Dirichlet[] PhiDist
    );

    /// <summary>
    /// Posterior distributions returned by inference, containing both the worker parameter
    /// posteriors and the inferred item-level distributions.
    /// </summary>
    /// <param name="ThetaDist">Posterior Beta distributions for each worker's spammer probability (theta).</param>
    /// <param name="PhiDist">Posterior Dirichlet distributions for each worker's label preferences when spamming (phi).</param>
    /// <param name="TDist">Posterior Discrete distributions over the true label for each item (T).</param>
    /// <param name="SDist">
    /// Posterior Bernoulli distributions for whether a worker was spamming on an item.
    /// Indexed by annotation slot, not by worker: <c>SDist[item][k]</c> corresponds to
    /// <c>SparseAnnotations.WorkerIndices[item][k]</c>.
    /// </param>
    public record ModelPosterior(
        Beta[] ThetaDist,
        Dirichlet[] PhiDist,
        Discrete[] TDist,
        Bernoulli[][] SDist
    ) : ModelPriors(ThetaDist, PhiDist);
}
