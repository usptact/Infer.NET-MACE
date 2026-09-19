using MACE.Online;

namespace MACE.Service
{
    /// <summary>Configuration for the inference service.</summary>
    public sealed class ServiceOptions
    {
        /// <summary>Configuration section these options are bound from.</summary>
        public const string SectionName = "Mace";

        /// <summary>
        /// Number of workers the service serves. Fixed at startup: worker indices are positions in
        /// the parameter arrays, so a roster that changed size would silently re-point every index.
        /// </summary>
        public int NumWorkers { get; set; } = 8;

        /// <summary>Number of label categories.</summary>
        public int NumCategories { get; set; } = 3;

        /// <summary>How many models to warm, which caps concurrent inference.</summary>
        public int PoolSize { get; set; } = 4;

        /// <summary>Number of EP inference iterations per call.</summary>
        public int Iterations { get; set; } = 50;

        /// <summary>
        /// RNG seed for label initialisation. Shared by every pooled model so the answer does not
        /// depend on which slot a request lands on.
        /// </summary>
        public int? Seed { get; set; } = 42;

        /// <summary>Default weight given to one item's evidence when feedback does not specify.</summary>
        public double LearningRate { get; set; } = PriorUpdateService.DefaultLearningRate;

        /// <summary>
        /// Default share of accumulated evidence that survives each update, in (0, 1].
        /// </summary>
        /// <remarks>
        /// The default forgets, unlike the library default, because a service runs indefinitely. At
        /// 0.99 with the default learning rate a worker's reliability reflects roughly the last 50
        /// items they were judged on, so a worker whose behaviour changes is re-learned instead of
        /// being held to their history forever. Set it to 1.0 to accumulate without limit.
        /// </remarks>
        public double Retention { get; set; } = 0.99;

        /// <summary>Path to load worker parameters from at startup, and save to at shutdown.</summary>
        /// <remarks>
        /// Null keeps everything in memory, which means reliability is lost on restart. Pointing this
        /// at a durable volume is what makes the feedback loop survive a redeploy.
        /// </remarks>
        public string? BeliefStorePath { get; set; }

        /// <summary>Throws when the configured values cannot produce a usable service.</summary>
        /// <exception cref="InvalidOperationException">Thrown when a value is out of range.</exception>
        public void Validate()
        {
            if (NumWorkers <= 0)
                throw new InvalidOperationException($"{SectionName}:NumWorkers must be positive.");
            if (NumCategories <= 0)
                throw new InvalidOperationException($"{SectionName}:NumCategories must be positive.");
            if (PoolSize <= 0)
                throw new InvalidOperationException($"{SectionName}:PoolSize must be positive.");
            if (Iterations <= 0)
                throw new InvalidOperationException($"{SectionName}:Iterations must be positive.");
            if (LearningRate <= 0.0)
                throw new InvalidOperationException($"{SectionName}:LearningRate must be positive.");
            if (Retention <= 0.0 || Retention > 1.0)
                throw new InvalidOperationException($"{SectionName}:Retention must be in (0, 1].");
        }
    }
}
