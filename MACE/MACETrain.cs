using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace MACE
{
    /// <summary>
    /// Implements the MACE (Multi-Annotator Competence Estimation) model for crowdsourcing annotation quality estimation.
    /// Uses a sparse jagged representation — only observed (item, worker, label) triples are stored,
    /// so memory and per-iteration cost scale with the number of annotations rather than the full item×worker matrix.
    /// </summary>
    public class MACETrain : MACEBase
    {
        // Number of observed annotations per item (varies per item)
        private VariableArray<int> _numObsPerItem;

        // Jagged range: obsRange[item] covers only the workers who annotated that item
        private Microsoft.ML.Probabilistic.Models.Range _obsRange;

        // Observed worker indices: _observedWorkerIndices[item][k] = worker index for k-th annotation of item
        private VariableArray<VariableArray<int>, int[][]> _observedWorkerIndices;

        // Observed labels: _observedLabels[item][k] = label given by _observedWorkerIndices[item][k]
        private VariableArray<VariableArray<int>, int[][]> _observedLabels;

        // Spammer indicators: _spammerIndicators[item][k] = whether the k-th annotator of item is spamming
        private VariableArray<VariableArray<bool>, bool[][]> _spammerIndicators;

        // Optional RNG seed for reproducible label initialisation
        private readonly int? _seed;

        /// <summary>
        /// Initializes a new instance of the MACETrain class and builds the probabilistic model.
        /// </summary>
        /// <param name="numWorkers">Number of workers in the dataset.</param>
        /// <param name="numItems">Number of items to be annotated.</param>
        /// <param name="numCategories">Number of possible label categories.</param>
        /// <param name="iterations">Number of VMP inference iterations (default: 50).</param>
        /// <param name="seed">Optional RNG seed for reproducible label initialisation. When null, results may vary across runs.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is non-positive.</exception>
        public MACETrain(int numWorkers, int numItems, int numCategories, int iterations = 50, int? seed = null)
        {
            if (numWorkers <= 0)
                throw new ArgumentOutOfRangeException(nameof(numWorkers), "Number of workers must be positive.");
            if (numItems <= 0)
                throw new ArgumentOutOfRangeException(nameof(numItems), "Number of items must be positive.");
            if (numCategories <= 0)
                throw new ArgumentOutOfRangeException(nameof(numCategories), "Number of categories must be positive.");
            if (iterations <= 0)
                throw new ArgumentOutOfRangeException(nameof(iterations), "Number of iterations must be positive.");

            _seed = seed;

            _numWorkers.ObservedValue = numWorkers;
            _numItems.ObservedValue = numItems;
            _numCategories.ObservedValue = numCategories;

            _numObsPerItem = Variable.Array<int>(_itemRange).Named("numObs");
            _obsRange = new Microsoft.ML.Probabilistic.Models.Range(_numObsPerItem[_itemRange]).Named("obs");

            _observedWorkerIndices = Variable.Array(Variable.Array<int>(_obsRange), _itemRange).Named("workerIdx");
            _observedLabels = Variable.Array(Variable.Array<int>(_obsRange), _itemRange).Named("label");
            _spammerIndicators = Variable.Array(Variable.Array<bool>(_obsRange), _itemRange).Named("S");

            CreateModel();
            InferenceEngine.NumberOfIterations = iterations;
        }

        /// <summary>
        /// Creates the complete MACE probabilistic model using a sparse jagged representation.
        /// The inner loop runs only over observed (item, worker) pairs — no sentinel values needed.
        /// </summary>
        protected override void CreateModel()
        {
            base.CreateModel();

            using (Variable.ForEach(_itemRange))
            {
                _trueLabels[_itemRange] = Variable.DiscreteUniform(_numCategories);

                using (Variable.ForEach(_obsRange))
                {
                    var workerIdx = _observedWorkerIndices[_itemRange][_obsRange];

                    _spammerIndicators[_itemRange][_obsRange] = Variable.Bernoulli(_theta[workerIdx]);

                    using (Variable.If(_spammerIndicators[_itemRange][_obsRange] == false))
                    {
                        _observedLabels[_itemRange][_obsRange] = _trueLabels[_itemRange];
                    }

                    using (Variable.If(_spammerIndicators[_itemRange][_obsRange] == true))
                    {
                        _observedLabels[_itemRange][_obsRange] = Variable.Discrete(_phi[workerIdx]);
                    }
                }
            }

            _observedLabels.AddAttribute(new DoNotInfer());
            _observedWorkerIndices.AddAttribute(new DoNotInfer());
        }

        /// <summary>
        /// Runs VMP inference and returns posterior distributions for all model parameters.
        /// The returned <see cref="ModelPosterior"/> can be passed directly as priors to a
        /// subsequent call to support incremental/online learning.
        /// SDist[item][k] is parallel to annotations.WorkerIndices[item][k].
        /// </summary>
        /// <param name="annotations">Sparse annotation data from <see cref="CsvReader.GetSparseData"/>.</param>
        /// <param name="priors">Prior distributions for worker parameters (theta and phi).</param>
        /// <returns>Posterior distributions for all model parameters.</returns>
        /// <exception cref="ArgumentNullException">Thrown when annotations or priors is null.</exception>
        /// <exception cref="ArgumentException">Thrown when annotation dimensions don't match the model.</exception>
        public ModelPosterior InferModelData(SparseAnnotations annotations, ModelPriors priors)
        {
            if (annotations == null)
                throw new ArgumentNullException(nameof(annotations));
            if (priors == null)
                throw new ArgumentNullException(nameof(priors));

            if (annotations.WorkerIndices.Length != _numItems.ObservedValue)
                throw new ArgumentException(
                    $"Annotations have {annotations.WorkerIndices.Length} items but model expects {_numItems.ObservedValue}.",
                    nameof(annotations));

            if (_seed.HasValue)
                Rand.Restart(_seed.Value);

            InitializeLabels(_numItems.ObservedValue, _numCategories.ObservedValue);
            SetModelData(priors);

            _numObsPerItem.ObservedValue = annotations.WorkerIndices.Select(w => w.Length).ToArray();
            _observedWorkerIndices.ObservedValue = annotations.WorkerIndices;
            _observedLabels.ObservedValue = annotations.Labels;

            return new ModelPosterior(
                ThetaDist: InferenceEngine.Infer<Beta[]>(_theta),
                PhiDist: InferenceEngine.Infer<Dirichlet[]>(_phi),
                TDist: InferenceEngine.Infer<Discrete[]>(_trueLabels),
                SDist: InferenceEngine.Infer<Bernoulli[][]>(_spammerIndicators)
            );
        }
    }
}
