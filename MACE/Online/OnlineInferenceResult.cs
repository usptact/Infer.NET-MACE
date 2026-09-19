using Microsoft.ML.Probabilistic.Distributions;

namespace MACE.Online
{
    /// <summary>
    /// Result of a single-item online inference call.
    /// </summary>
    /// <remarks>
    /// The batch API returns arrays indexed by item, which a caller inferring one item at a time
    /// would have to index with a literal <c>[0]</c> everywhere. This unwraps that, and carries the
    /// worker indices alongside the spammer posteriors so the two stay interpretable together.
    /// </remarks>
    /// <param name="LabelDist">Posterior over the item's true label.</param>
    /// <param name="Label">Most probable label — <c>argmax(LabelDist)</c>.</param>
    /// <param name="Confidence">Probability of <paramref name="Label"/>.</param>
    /// <param name="Entropy">
    /// Shannon entropy of <paramref name="LabelDist"/> in nats, from 0 (certain) to ln(numCategories)
    /// (no information). Confidence alone cannot distinguish "one strong runner-up" from "spread
    /// evenly across the rest", which matters when deciding what to escalate for review.
    /// </param>
    /// <param name="ContributingWorkers">
    /// Indices of the workers who actually annotated this item, in the order their posteriors appear
    /// in <paramref name="SpammerDist"/>.
    /// </param>
    /// <param name="SpammerDist">
    /// Per-annotation spammer posteriors. Parallel to <paramref name="ContributingWorkers"/>, not
    /// indexed by worker id — absent workers have no entry.
    /// </param>
    public record OnlineInferenceResult(
        Discrete LabelDist,
        int Label,
        double Confidence,
        double Entropy,
        int[] ContributingWorkers,
        Bernoulli[] SpammerDist
    )
    {
        /// <summary>
        /// Spammer probability for a specific worker, or null when that worker did not annotate
        /// this item. Saves callers from walking <see cref="ContributingWorkers"/> themselves.
        /// </summary>
        /// <param name="worker">Worker index, as used by the model.</param>
        public double? SpammerProbabilityFor(int worker)
        {
            for (int k = 0; k < ContributingWorkers.Length; k++)
            {
                if (ContributingWorkers[k] == worker)
                {
                    return SpammerDist[k].GetProbTrue();
                }
            }

            return null;
        }
    }
}
