using System.Diagnostics;
using Grpc.Core;
using MACE;
using MACE.Online;
using MACE.Service.Grpc;
using Microsoft.Extensions.Options;
using Prometheus;

namespace MACE.Service.Services
{
    /// <summary>
    /// gRPC surface over <see cref="MACETrain.InferOnline"/> and <see cref="PriorUpdateService"/>.
    /// </summary>
    /// <remarks>
    /// Inference and feedback are deliberately separate calls. Inference reads worker parameters and
    /// never writes them, so it stays cheap and side-effect free; feedback writes them only when a
    /// true label is actually known. Folding the two together would let a single unreviewed item
    /// move a worker's reliability.
    /// </remarks>
    public sealed class MaceInferenceService : MaceInference.MaceInferenceBase
    {
        private static readonly Counter InferenceRequests = Metrics.CreateCounter(
            "mace_inference_requests_total", "Label inference requests.", "outcome");

        private static readonly Histogram InferenceDuration = Metrics.CreateHistogram(
            "mace_inference_duration_seconds", "Time spent inferring a label, excluding pool wait.");

        private static readonly Histogram PoolWaitDuration = Metrics.CreateHistogram(
            "mace_inference_pool_wait_seconds", "Time spent waiting for a free model.");

        private static readonly Counter FeedbackUpdates = Metrics.CreateCounter(
            "mace_feedback_workers_updated_total", "Worker parameter updates applied from feedback.");

        private static readonly Gauge PoolAvailable = Metrics.CreateGauge(
            "mace_inference_pool_available", "Models not currently leased.");

        private readonly IInferencePool _pool;
        private readonly BeliefStore _beliefs;
        private readonly ServiceOptions _options;
        private readonly ILogger<MaceInferenceService> _logger;

        /// <summary>Creates the service.</summary>
        /// <param name="pool">Pool of warmed single-item models.</param>
        /// <param name="beliefs">Store holding current worker parameters.</param>
        /// <param name="options">Service configuration.</param>
        /// <param name="logger">Destination for per-call diagnostics.</param>
        public MaceInferenceService(
            IInferencePool pool,
            BeliefStore beliefs,
            IOptions<ServiceOptions> options,
            ILogger<MaceInferenceService> logger)
        {
            _pool = pool;
            _beliefs = beliefs;
            _options = options.Value;
            _logger = logger;
        }

        /// <inheritdoc />
        public override async Task<InferLabelResponse> InferLabel(
            InferLabelRequest request, ServerCallContext context)
        {
            var annotations = ToAnnotationArray(request.Annotations);

            var waited = Stopwatch.StartNew();
            using var lease = await _pool.AcquireAsync(context.CancellationToken);
            waited.Stop();
            PoolWaitDuration.Observe(waited.Elapsed.TotalSeconds);
            PoolAvailable.Set(_pool.Available);

            OnlineInferenceResult result;
            var inferring = Stopwatch.StartNew();
            try
            {
                result = lease.Model.InferOnline(annotations, _beliefs.Snapshot());
            }
            catch (Exception ex)
            {
                InferenceRequests.WithLabels("error").Inc();
                _logger.LogError(ex, "Inference failed for item {ItemId}", request.ItemId);
                throw new RpcException(new Status(StatusCode.Internal, "Inference failed."));
            }
            finally
            {
                inferring.Stop();
                InferenceDuration.Observe(inferring.Elapsed.TotalSeconds);
                PoolAvailable.Set(_pool.Available);
            }

            InferenceRequests.WithLabels("ok").Inc();

            if (result.ContributingWorkers.Length == 0)
            {
                _logger.LogWarning(
                    "Item {ItemId} had no annotations; the returned label is the prior and carries no evidence.",
                    request.ItemId);
            }

            var response = new InferLabelResponse
            {
                ItemId = request.ItemId,
                Label = result.Label,
                Confidence = result.Confidence,
                Entropy = result.Entropy
            };

            response.LabelProbabilities.AddRange(result.LabelDist.GetProbs());
            response.ContributingWorkers.AddRange(result.ContributingWorkers);

            for (int k = 0; k < result.ContributingWorkers.Length; k++)
            {
                response.WorkerAssessments.Add(new WorkerAssessment
                {
                    Worker = result.ContributingWorkers[k],
                    SpammerProbability = result.SpammerDist[k].GetProbTrue()
                });
            }

            _logger.LogDebug(
                "Item {ItemId}: label {Label} at {Confidence:F3} confidence, entropy {Entropy:F3}, "
                + "from {Count} annotation(s) in {Ms}ms",
                request.ItemId, result.Label, result.Confidence, result.Entropy,
                result.ContributingWorkers.Length, inferring.ElapsedMilliseconds);

            return response;
        }

        /// <inheritdoc />
        public override Task<SubmitFeedbackResponse> SubmitFeedback(
            SubmitFeedbackRequest request, ServerCallContext context)
        {
            var annotations = ToAnnotationArray(request.Annotations);

            if (request.TrueLabel < 0 || request.TrueLabel >= _options.NumCategories)
            {
                throw new RpcException(new Status(
                    StatusCode.InvalidArgument,
                    $"true_label {request.TrueLabel} is outside [0, {_options.NumCategories})."));
            }

            double learningRate = request.HasLearningRate ? request.LearningRate : _options.LearningRate;
            double retention = request.HasRetention ? request.Retention : _options.Retention;

            if (learningRate <= 0.0)
            {
                throw new RpcException(new Status(
                    StatusCode.InvalidArgument, "learning_rate must be positive."));
            }

            if (retention <= 0.0 || retention > 1.0)
            {
                throw new RpcException(new Status(
                    StatusCode.InvalidArgument, "retention must be in (0, 1]."));
            }

            var (priors, updatedWorkers) = _beliefs.ApplyFeedback(
                annotations, request.TrueLabel, learningRate, retention);

            FeedbackUpdates.Inc(updatedWorkers.Length);

            var response = new SubmitFeedbackResponse { ItemId = request.ItemId };
            foreach (int worker in updatedWorkers)
            {
                response.Updated.Add(Describe(priors, worker));
            }

            _logger.LogInformation(
                "Feedback on item {ItemId}: true label {TrueLabel} moved {Count} worker(s) "
                + "(learningRate {LearningRate}, retention {Retention})",
                request.ItemId, request.TrueLabel, updatedWorkers.Length, learningRate, retention);

            return Task.FromResult(response);
        }

        /// <inheritdoc />
        public override Task<GetWorkerReliabilityResponse> GetWorkerReliability(
            GetWorkerReliabilityRequest request, ServerCallContext context)
        {
            var priors = _beliefs.Snapshot();
            var response = new GetWorkerReliabilityResponse();

            var requested = request.Workers.Count > 0
                ? request.Workers.ToArray()
                : Enumerable.Range(0, priors.ThetaDist.Length).ToArray();

            foreach (int worker in requested)
            {
                if (worker < 0 || worker >= priors.ThetaDist.Length)
                {
                    throw new RpcException(new Status(
                        StatusCode.InvalidArgument,
                        $"Worker {worker} is outside [0, {priors.ThetaDist.Length})."));
                }

                response.Workers.Add(Describe(priors, worker));
            }

            return Task.FromResult(response);
        }

        private static WorkerReliability Describe(ModelPriors priors, int worker)
        {
            var theta = priors.ThetaDist[worker];
            var reliability = new WorkerReliability
            {
                Worker = worker,
                SpammerProbability = theta.GetMean(),

                // Total pseudo-counts less the uninformative prior: what the worker has actually
                // earned, rather than what they started with.
                Evidence = theta.TrueCount + theta.FalseCount - 2.0
            };

            reliability.SpamPreferences.AddRange(priors.PhiDist[worker].GetMean());
            return reliability;
        }

        /// <summary>
        /// Converts sparse (worker, label) pairs into the dense per-worker array the model takes,
        /// rejecting anything that would silently mean something other than the caller intended.
        /// </summary>
        private int[] ToAnnotationArray(IEnumerable<Annotation> annotations)
        {
            var dense = Enumerable.Repeat(MACETrain.MissingAnnotation, _options.NumWorkers).ToArray();

            foreach (var annotation in annotations)
            {
                if (annotation.Worker < 0 || annotation.Worker >= _options.NumWorkers)
                {
                    throw new RpcException(new Status(
                        StatusCode.InvalidArgument,
                        $"Worker {annotation.Worker} is outside [0, {_options.NumWorkers})."));
                }

                if (annotation.Label < 0 || annotation.Label >= _options.NumCategories)
                {
                    throw new RpcException(new Status(
                        StatusCode.InvalidArgument,
                        $"Label {annotation.Label} from worker {annotation.Worker} is outside "
                        + $"[0, {_options.NumCategories})."));
                }

                // The model takes one annotation per worker per item. Silently keeping the last one
                // would make the result depend on request ordering.
                if (dense[annotation.Worker] != MACETrain.MissingAnnotation)
                {
                    throw new RpcException(new Status(
                        StatusCode.InvalidArgument,
                        $"Worker {annotation.Worker} appears more than once in this request."));
                }

                dense[annotation.Worker] = annotation.Label;
            }

            return dense;
        }
    }
}
