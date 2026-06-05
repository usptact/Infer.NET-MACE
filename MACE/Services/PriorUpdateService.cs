namespace MACE.Services;

public enum Verdict { TrueAlarm, FalseAlarm }

public record BetaParameters(double Alpha, double Beta);

/// <summary>
/// Computes updated Beta priors from operator feedback.
///
/// This is pure arithmetic — no Infer.NET required. The update rules follow
/// THREATSENSE_DESIGN.md §7.5 and interpret the spammer posterior S[0][j]
/// produced by a previous Infer call.
/// </summary>
public sealed class PriorUpdateService
{
    // Annotations >= this value are treated as "flagged a threat"
    private const int MediumThreshold = 2;

    /// <summary>
    /// Returns a new <see cref="BetaParameters"/> with the Beta distribution
    /// updated based on whether the sensor's behaviour was consistent with the verdict.
    /// </summary>
    /// <param name="current">Current Beta(α, β) prior for sensor j.</param>
    /// <param name="spammerProbMean">
    ///   Mean of S[0][j] from the most recent Infer call on this incident.
    /// </param>
    /// <param name="annotation">
    ///   Discretised label sensor j provided. -1 if sensor was absent.
    /// </param>
    /// <param name="verdict">Operator's verdict on the closed incident.</param>
    /// <param name="learningRate">
    ///   Controls how aggressively one incident shifts the prior. Default: 0.5.
    /// </param>
    public BetaParameters UpdateTheta(
        BetaParameters current,
        double faultProbMean,
        int annotation,
        Verdict verdict,
        double learningRate = 0.5)
    {
        if (annotation == -1)
            return current;  // sensor absent — no evidence either way

        double alpha = current.Alpha;
        double beta  = current.Beta;
        bool flaggedThreat = annotation >= MediumThreshold;

        if (verdict == Verdict.TrueAlarm)
        {
            if (flaggedThreat)
                // Sensor correctly flagged → reinforce reliability (increase β)
                beta  += learningRate * (1.0 - faultProbMean);
            else
                // Sensor missed a real threat → penalise slightly (increase α)
                alpha += learningRate * 0.3;
        }
        else  // FalseAlarm
        {
            if (flaggedThreat)
                // Sensor contributed to false alarm → penalise (increase α)
                alpha += learningRate * faultProbMean;
            else
                // Sensor correctly stayed quiet → reinforce (increase β)
                beta  += learningRate * 0.3;
        }

        return new BetaParameters(alpha, beta);
    }

    public static Verdict ParseVerdict(string raw) => raw.ToUpperInvariant() switch
    {
        "TRUE_ALARM"  => Verdict.TrueAlarm,
        "FALSE_ALARM" => Verdict.FalseAlarm,
        _             => throw new ArgumentException($"Unknown verdict '{raw}'. Expected TRUE_ALARM or FALSE_ALARM.")
    };
}
