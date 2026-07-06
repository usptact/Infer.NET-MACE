namespace MACE.Services;

public enum Verdict { TrueAlarm, FalseAlarm }

public record BetaParameters(double Alpha, double Beta);

/// <summary>
/// Updated θ (Beta) and φ (Dirichlet) priors for a single sensor type.
/// </summary>
public record BeliefUpdate(BetaParameters Theta, double[] Phi);

/// <summary>
/// Computes updated Beta (θ) and Dirichlet (φ) priors from operator feedback.
///
/// This is pure arithmetic — no Infer.NET required. It implements the
/// single-incident EM step of the MACE generative model (see
/// <see cref="MACETrain.CreateModel"/>): once the operator resolves an incident
/// to a gold true level, the fault responsibility of each sensor is available in
/// closed form, and the conjugate θ / φ updates follow directly.
///
/// Generative model recap for sensor j on this incident:
///   S ~ Bernoulli(θ)                         // S = 1 ⇒ faulty
///   reading = trueLevel        if S = 0      // reliable sensor reports the truth
///   reading ~ Categorical(φ)   if S = 1      // faulty sensor draws from its bias
///
/// Given the gold <c>trueLevel</c> and the observed <c>reading</c>, the fault
/// responsibility is
///   r = P(S = 1 | reading, trueLevel)
///     = 1                                            if reading ≠ trueLevel
///     = θ̄·φ̄[reading] / (θ̄·φ̄[reading] + (1 − θ̄))     if reading = trueLevel
/// where θ̄, φ̄ are the current prior means. The matching conjugate updates are
///   θ:  α += r,   β += (1 − r)
///   φ:  pseudocounts[reading] += r          // only the faulty branch informs φ
/// each scaled by the learning rate.
/// </summary>
public sealed class PriorUpdateService
{
    /// <summary>
    /// Returns updated θ and φ priors for one sensor type from a single resolved incident.
    /// </summary>
    /// <param name="theta">Current Beta(α, β) prior for the sensor's fault rate.</param>
    /// <param name="phi">
    ///   Current Dirichlet pseudocounts for the sensor's fault-bias distribution.
    ///   Length must equal the number of threat levels.
    /// </param>
    /// <param name="reading">Discretised label the sensor produced. -1 if the sensor was absent.</param>
    /// <param name="trueLevel">Operator-resolved gold threat level for the incident.</param>
    /// <param name="learningRate">
    ///   Damping factor applied to the conjugate counts. 1.0 is the undamped EM step;
    ///   values in (0, 1) down-weight a single incident's evidence. Default: 0.5.
    /// </param>
    public BeliefUpdate UpdateBeliefs(
        BetaParameters theta,
        double[] phi,
        int reading,
        int trueLevel,
        double learningRate = 0.5)
    {
        ArgumentNullException.ThrowIfNull(phi);

        // Sensor absent — no evidence either way; leave both priors untouched.
        if (reading == -1)
            return new BeliefUpdate(theta, phi);

        if (reading < 0 || reading >= phi.Length)
            throw new ArgumentOutOfRangeException(nameof(reading),
                $"reading {reading} is outside [0, {phi.Length}).");

        // Fault responsibility r = P(faulty | reading, trueLevel).
        double r;
        if (reading != trueLevel)
        {
            // A reliable sensor must report the truth; it did not ⇒ certainly faulty.
            r = 1.0;
        }
        else
        {
            // reading == trueLevel: the reading is consistent with either a reliable
            // sensor (reports truth) or a faulty one that happened to emit the truth.
            double thetaMean = theta.Alpha / (theta.Alpha + theta.Beta);
            double phiSum    = 0.0;
            for (int i = 0; i < phi.Length; i++) phiSum += phi[i];
            double phiMean   = phi[reading] / phiSum;

            double faulty   = thetaMean * phiMean;
            double reliable = 1.0 - thetaMean;   // indicator[reading == trueLevel] = 1 here
            r = faulty / (faulty + reliable);    // denominator ≥ reliable > 0 for α, β > 0
        }

        // Conjugate updates, damped by the learning rate.
        double alpha = theta.Alpha + learningRate * r;
        double beta  = theta.Beta  + learningRate * (1.0 - r);

        double[] newPhi = (double[])phi.Clone();
        newPhi[reading] += learningRate * r;   // only the faulty branch, only bucket 'reading'

        return new BeliefUpdate(new BetaParameters(alpha, beta), newPhi);
    }

    public static Verdict ParseVerdict(string raw) => raw.ToUpperInvariant() switch
    {
        "TRUE_ALARM"  => Verdict.TrueAlarm,
        "FALSE_ALARM" => Verdict.FalseAlarm,
        _             => throw new ArgumentException($"Unknown verdict '{raw}'. Expected TRUE_ALARM or FALSE_ALARM.")
    };
}
