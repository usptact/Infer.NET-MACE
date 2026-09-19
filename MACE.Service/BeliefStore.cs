using MACE;
using MACE.Online;

namespace MACE.Service
{
    /// <summary>
    /// Holds the current worker parameters for the service.
    /// </summary>
    /// <remarks>
    /// Inference needs the parameters as they are now; feedback needs to move them. Both arrive on
    /// arbitrary gRPC threads, so the two have to be serialised against each other: without that,
    /// two feedback calls that read the same starting point would each write back their own result
    /// and one item's evidence would vanish.
    ///
    /// <para>
    /// Reads take a snapshot rather than the live reference. <see cref="ModelPriors"/> holds arrays,
    /// so handing out the live instance would let a caller observe a half-applied update. Updates
    /// replace the whole object instead of mutating it, which keeps every snapshot internally
    /// consistent for as long as its holder needs it.
    /// </para>
    ///
    /// <para>
    /// This keeps state in memory, which means it is lost on restart and not shared between
    /// replicas. <see cref="ModelPriorsIo"/> gives it somewhere durable to live; wiring that to real
    /// storage is a deployment decision, so it is deliberately left to the host.
    /// </para>
    /// </remarks>
    public sealed class BeliefStore
    {
        private readonly Lock _gate = new();
        private readonly PriorUpdateService _updates;
        private ModelPriors _current;

        /// <summary>Creates a store over an initial set of parameters.</summary>
        /// <param name="initial">Starting worker parameters.</param>
        /// <param name="updates">The feedback rule used by <see cref="ApplyFeedback"/>.</param>
        /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
        public BeliefStore(ModelPriors initial, PriorUpdateService updates)
        {
            ArgumentNullException.ThrowIfNull(initial);
            ArgumentNullException.ThrowIfNull(updates);

            _current = initial;
            _updates = updates;
        }

        /// <summary>Number of workers these parameters describe.</summary>
        public int NumWorkers => _current.ThetaDist.Length;

        /// <summary>Number of label categories these parameters describe.</summary>
        public int NumCategories => _current.PhiDist.Length > 0 ? _current.PhiDist[0].Dimension : 0;

        /// <summary>Returns the parameters as they are now.</summary>
        public ModelPriors Snapshot()
        {
            lock (_gate)
            {
                return _current;
            }
        }

        /// <summary>
        /// Moves worker parameters using one resolved item, and reports which workers changed.
        /// </summary>
        /// <param name="annotations">
        /// One entry per worker, using <see cref="MACETrain.MissingAnnotation"/> for workers who did
        /// not annotate the item.
        /// </param>
        /// <param name="trueLabel">The established true label.</param>
        /// <param name="learningRate">Weight given to this item's evidence.</param>
        /// <param name="retention">Share of accumulated evidence that survives.</param>
        /// <returns>
        /// The updated parameters and the indices of the workers the feedback moved, which is the
        /// subset a caller needs to see rather than the whole roster.
        /// </returns>
        public (ModelPriors Priors, int[] UpdatedWorkers) ApplyFeedback(
            int[] annotations,
            int trueLabel,
            double learningRate,
            double retention)
        {
            ArgumentNullException.ThrowIfNull(annotations);

            lock (_gate)
            {
                _current = _updates.UpdateFromResolvedItem(
                    _current, annotations, trueLabel, learningRate, retention);

                var moved = new List<int>();
                for (int worker = 0; worker < annotations.Length; worker++)
                {
                    if (annotations[worker] != MACETrain.MissingAnnotation)
                    {
                        moved.Add(worker);
                    }
                }

                return (_current, moved.ToArray());
            }
        }

        /// <summary>
        /// Replaces the stored parameters wholesale, for loading persisted state at startup.
        /// </summary>
        /// <param name="priors">Parameters to store.</param>
        /// <exception cref="ArgumentNullException">Thrown when priors is null.</exception>
        /// <exception cref="ArgumentException">Thrown when the shape differs from what is stored.</exception>
        public void Replace(ModelPriors priors)
        {
            ArgumentNullException.ThrowIfNull(priors);

            lock (_gate)
            {
                if (priors.ThetaDist.Length != _current.ThetaDist.Length)
                {
                    throw new ArgumentException(
                        $"Replacement describes {priors.ThetaDist.Length} workers but the store holds "
                        + $"{_current.ThetaDist.Length}.",
                        nameof(priors));
                }

                _current = priors;
            }
        }
    }
}
