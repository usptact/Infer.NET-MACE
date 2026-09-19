using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace MACE.Online
{
    /// <summary>Beta parameters for one worker's spammer rate.</summary>
    /// <param name="Alpha">Pseudo-count of spamming behaviour.</param>
    /// <param name="Beta">Pseudo-count of honest behaviour.</param>
    public record BetaParameters(double Alpha, double Beta);

    /// <summary>Updated spammer rate and label preferences for one worker.</summary>
    /// <param name="Theta">Updated Beta parameters for the worker's spammer rate.</param>
    /// <param name="Phi">Updated Dirichlet pseudo-counts for the worker's spam label preferences.</param>
    public record BeliefUpdate(BetaParameters Theta, double[] Phi);

    /// <summary>
    /// Moves worker parameters from items whose true label has been established.
    /// </summary>
    /// <remarks>
    /// This is the supervised counterpart to batch inference. Where the batch model estimates worker
    /// reliability and true labels together from annotations alone, this takes a known true label and
    /// updates the workers who annotated that item. It is closed-form arithmetic with no Infer.NET
    /// involvement, which is what makes it cheap enough to run on every resolution.
    ///
    /// <para>The generative model for worker j on one item:</para>
    /// <code>
    /// S ~ Bernoulli(theta)                  // S = 1 means spamming
    /// label = trueLabel        if S = 0     // an honest worker reports the truth
    /// label ~ Categorical(phi) if S = 1     // a spammer draws from its own preference
    /// </code>
    ///
    /// <para>
    /// Given the true label and the worker's label, the spam responsibility follows directly:
    /// </para>
    /// <code>
    /// r = P(S = 1 | label, trueLabel)
    ///   = 1                                                if label != trueLabel
    ///   = theta*phi[label] / (theta*phi[label] + (1-theta)) if label == trueLabel
    /// </code>
    ///
    /// <para>
    /// A worker who disagreed with an established truth cannot have been honest under this model, so
    /// r is exactly 1. Agreement is weaker evidence, because a spammer can produce the right label by
    /// chance, and r apportions the credit accordingly. The conjugate updates are then
    /// <c>alpha += r</c>, <c>beta += 1 - r</c>, and <c>phi[label] += r</c> — only the spamming branch
    /// informs phi, since that is the only branch phi explains.
    /// </para>
    /// </remarks>
    public sealed class PriorUpdateService
    {
        /// <summary>Weight applied to one item's evidence when no other value is given.</summary>
        public const double DefaultLearningRate = 0.5;

        /// <summary>Retention applied to accumulated evidence when no other value is given: 1.0 keeps everything.</summary>
        public const double DefaultRetention = 1.0;

        /// <summary>
        /// Updates one worker's parameters from one item whose true label is known.
        /// </summary>
        /// <param name="theta">The worker's current spammer-rate parameters.</param>
        /// <param name="phi">
        /// The worker's current spam label preferences, as Dirichlet pseudo-counts.
        /// Length must equal the number of categories.
        /// </param>
        /// <param name="label">
        /// The label this worker gave, or <see cref="MACETrain.MissingAnnotation"/> if they did not
        /// annotate the item, in which case both parameters are returned unchanged.
        /// </param>
        /// <param name="trueLabel">The established true label for the item.</param>
        /// <param name="learningRate">
        /// Weight given to this item's evidence. 1.0 is the undamped step; smaller values stop a
        /// single item from moving a worker far.
        /// </param>
        /// <param name="retention">
        /// How much accumulated evidence survives this update, in (0, 1]. 1.0 never forgets, which
        /// means a worker's parameters harden over time and a worker whose behaviour changes can
        /// never be re-learned. Below 1.0, evidence decays geometrically and the accumulated counts
        /// settle at a finite level — see <see cref="EffectiveSampleSize"/>.
        /// </param>
        /// <returns>The updated parameters. Inputs are not mutated.</returns>
        /// <exception cref="ArgumentNullException">Thrown when phi is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when the label is outside the category range, or the rates are outside their valid ranges.
        /// </exception>
        public BeliefUpdate UpdateBeliefs(
            BetaParameters theta,
            double[] phi,
            int label,
            int trueLabel,
            double learningRate = DefaultLearningRate,
            double retention = DefaultRetention)
        {
            ArgumentNullException.ThrowIfNull(theta);
            ArgumentNullException.ThrowIfNull(phi);

            if (learningRate <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
            if (retention <= 0.0 || retention > 1.0)
                throw new ArgumentOutOfRangeException(nameof(retention), "Retention must be in (0, 1].");

            // A worker who did not annotate this item gives no evidence either way. Returning early
            // matters: applying retention here would let absence itself erode their history.
            if (label == MACETrain.MissingAnnotation)
            {
                return new BeliefUpdate(theta, phi);
            }

            if (label < 0 || label >= phi.Length)
                throw new ArgumentOutOfRangeException(nameof(label),
                    $"Label {label} is outside [0, {phi.Length}).");
            if (trueLabel < 0 || trueLabel >= phi.Length)
                throw new ArgumentOutOfRangeException(nameof(trueLabel),
                    $"True label {trueLabel} is outside [0, {phi.Length}).");

            double responsibility = SpamResponsibility(theta, phi, label, trueLabel);

            // Decay towards the uninformative prior rather than towards zero: pseudo-counts below 1
            // would make the distributions improper.
            double alpha = Decay(theta.Alpha, retention) + learningRate * responsibility;
            double beta = Decay(theta.Beta, retention) + learningRate * (1.0 - responsibility);

            var updatedPhi = new double[phi.Length];
            for (int category = 0; category < phi.Length; category++)
            {
                updatedPhi[category] = Decay(phi[category], retention);
            }

            updatedPhi[label] += learningRate * responsibility;

            return new BeliefUpdate(new BetaParameters(alpha, beta), updatedPhi);
        }

        /// <summary>
        /// Applies <see cref="UpdateBeliefs"/> to every worker who annotated a resolved item.
        /// </summary>
        /// <param name="priors">Current parameters for all workers.</param>
        /// <param name="annotations">
        /// One entry per worker, in worker-index order, using <see cref="MACETrain.MissingAnnotation"/>
        /// for workers who did not annotate the item — the same shape
        /// <see cref="MACETrain.InferOnline"/> takes.
        /// </param>
        /// <param name="trueLabel">The established true label for the item.</param>
        /// <param name="learningRate">Weight given to this item's evidence.</param>
        /// <param name="retention">How much accumulated evidence survives this update.</param>
        /// <returns>Updated priors for all workers. The input is not mutated.</returns>
        /// <exception cref="ArgumentNullException">Thrown when priors or annotations is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the annotation array does not match the priors.</exception>
        public ModelPriors UpdateFromResolvedItem(
            ModelPriors priors,
            int[] annotations,
            int trueLabel,
            double learningRate = DefaultLearningRate,
            double retention = DefaultRetention)
        {
            ArgumentNullException.ThrowIfNull(priors);
            ArgumentNullException.ThrowIfNull(annotations);

            if (annotations.Length != priors.ThetaDist.Length)
                throw new ArgumentException(
                    $"Expected one entry per worker ({priors.ThetaDist.Length}) but got {annotations.Length}.",
                    nameof(annotations));

            var theta = new Beta[priors.ThetaDist.Length];
            var phi = new Dirichlet[priors.PhiDist.Length];

            for (int worker = 0; worker < priors.ThetaDist.Length; worker++)
            {
                var currentTheta = new BetaParameters(
                    priors.ThetaDist[worker].TrueCount,
                    priors.ThetaDist[worker].FalseCount);

                var currentPhi = priors.PhiDist[worker].PseudoCount.ToArray();

                var updated = UpdateBeliefs(
                    currentTheta, currentPhi, annotations[worker], trueLabel, learningRate, retention);

                theta[worker] = new Beta(updated.Theta.Alpha, updated.Theta.Beta);

                var counts = Vector.Zero(updated.Phi.Length);
                for (int category = 0; category < updated.Phi.Length; category++)
                {
                    counts[category] = updated.Phi[category];
                }

                phi[worker] = new Dirichlet(counts);
            }

            return new ModelPriors(theta, phi);
        }

        /// <summary>
        /// The amount of evidence a worker's parameters retain in the long run under a given
        /// learning rate and retention, expressed as a count of items.
        /// </summary>
        /// <remarks>
        /// Repeated updates add <c>learningRate</c> and keep a <c>retention</c> share of what came
        /// before, so the accumulated excess over the uninformative prior converges to
        /// <c>learningRate / (1 - retention)</c>. That number is the memory of the system: it says
        /// how many recent items a worker's reliability effectively reflects, and it is what makes a
        /// long-running deployment able to notice that a worker has changed.
        /// </remarks>
        /// <param name="learningRate">Weight given to each item's evidence.</param>
        /// <param name="retention">How much accumulated evidence survives each update.</param>
        /// <returns>The steady-state evidence count, or positive infinity when nothing is ever forgotten.</returns>
        public static double EffectiveSampleSize(
            double learningRate = DefaultLearningRate,
            double retention = DefaultRetention)
        {
            if (retention >= 1.0)
            {
                return double.PositiveInfinity;
            }

            return learningRate / (1.0 - retention);
        }

        /// <summary>
        /// P(worker was spamming | their label, the true label) under the MACE generative model.
        /// </summary>
        private static double SpamResponsibility(BetaParameters theta, double[] phi, int label, int trueLabel)
        {
            if (label != trueLabel)
            {
                // An honest worker reports the truth by definition, so disagreement settles it.
                return 1.0;
            }

            double thetaMean = theta.Alpha / (theta.Alpha + theta.Beta);

            double phiTotal = 0.0;
            for (int category = 0; category < phi.Length; category++)
            {
                phiTotal += phi[category];
            }

            double phiMean = phi[label] / phiTotal;

            double spamming = thetaMean * phiMean;
            double honest = 1.0 - thetaMean;

            // honest > 0 whenever alpha and beta are positive, so the denominator cannot vanish.
            return spamming / (spamming + honest);
        }

        /// <summary>Shrinks a pseudo-count towards 1, the uninformative prior, never below it.</summary>
        private static double Decay(double count, double retention)
            => 1.0 + retention * (count - 1.0);
    }
}
