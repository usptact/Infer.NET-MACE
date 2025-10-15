using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace MACE
{
    /// <summary>
    /// Implements the MACE (Multi-Annotator Competence Estimation) model for crowdsourcing annotation quality estimation.
    /// This class extends MACEBase to define the complete probabilistic model including the observation model.
    /// </summary>
    public class MACETrain : MACEBase
    {
        /// <summary>
        /// Worker-item annotation matrix containing the observed votes (partially observed).
        /// A[item][worker] contains the annotation for that item-worker pair, or -1 for missing annotations.
        /// </summary>
        protected VariableArray<VariableArray<int>, int[][]> _annotations;

        /// <summary>
        /// Initializes a new instance of the MACETrain class.
        /// </summary>
        /// <param name="numWorkers">Number of workers in the dataset.</param>
        /// <param name="numItems">Number of items to be annotated.</param>
        /// <param name="numCategories">Number of possible label categories.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is non-positive.</exception>
        public MACETrain(int numWorkers, int numItems, int numCategories)
        {
            if (numWorkers <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numWorkers), "Number of workers must be positive.");
            }

            if (numItems <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numItems), "Number of items must be positive.");
            }

            if (numCategories <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(numCategories), "Number of categories must be positive.");
            }

            _annotations = Variable.Array(Variable.Array<int>(_workerRange), _itemRange);

            _numWorkers.ObservedValue = numWorkers;
            _numItems.ObservedValue = numItems;
            _numCategories.ObservedValue = numCategories;
        }

        /// <summary>
        /// Creates the complete MACE probabilistic model including the observation model.
        /// </summary>
        public override void CreateModel()
        {
            base.CreateModel();

            using (Variable.ForEach(_itemRange))
            {
                // True label for this item (uniform prior over categories)
                _trueLabels[_itemRange] = Variable.DiscreteUniform(_numCategories);
                
                using (Variable.ForEach(_workerRange))
                {
                    // Spammer indicator for this worker-item pair
                    _spammerIndicators[_itemRange][_workerRange] = Variable.Bernoulli(_theta[_workerRange]);
                    
                    // Only process observed annotations (skip missing data marked with -1)
                    using (Variable.If(_annotations[_itemRange][_workerRange] > -1))
                    {
                        using (Variable.If(_spammerIndicators[_itemRange][_workerRange] == false))
                        {
                            // Not a spammer: assign the true label
                            _annotations[_itemRange][_workerRange] = _trueLabels[_itemRange];
                        }
                        
                        using (Variable.If(_spammerIndicators[_itemRange][_workerRange] == true))
                        {
                            // Spammer: assign label according to their preference distribution
                            _annotations[_itemRange][_workerRange] = Variable.Discrete(_phi[_workerRange]);
                        }
                    }
                }
            }

            // Prevent the inference engine from trying to infer the observed annotations
            // The annotations matrix can contain -1 values which are out of the domain
            _annotations.AddAttribute(new DoNotInfer());
        }

        /// <summary>
        /// Performs probabilistic inference to estimate the posterior distributions of all model parameters.
        /// </summary>
        /// <param name="data">The annotation data matrix where data[item][worker] gives the annotation or -1 for missing.</param>
        /// <returns>ModelData containing the posterior distributions for all model parameters.</returns>
        /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
        /// <exception cref="ArgumentException">Thrown when data dimensions don't match the expected dimensions.</exception>
        public ModelData InferModelData(int[][] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            // Validate data dimensions
            if (data.Length != _numItems.ObservedValue)
            {
                throw new ArgumentException(
                    $"Data has {data.Length} items but model expects {_numItems.ObservedValue} items.", 
                    nameof(data));
            }

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == null || data[i].Length != _numWorkers.ObservedValue)
                {
                    throw new ArgumentException(
                        $"Data row {i} has {data[i]?.Length ?? 0} workers but model expects {_numWorkers.ObservedValue} workers.", 
                        nameof(data));
                }
            }

            var posteriors = new ModelData();

            // Set the observed annotation data
            _annotations.ObservedValue = data;

            // Perform inference to get posterior distributions
            posteriors.ThetaDist = InferenceEngine.Infer<Beta[]>(_theta);
            posteriors.PhiDist = InferenceEngine.Infer<Dirichlet[]>(_phi);
            posteriors.TDist = InferenceEngine.Infer<Discrete[]>(_trueLabels);
            posteriors.SDist = InferenceEngine.Infer<Bernoulli[][]>(_spammerIndicators);

            return posteriors;
        }
    }
}
