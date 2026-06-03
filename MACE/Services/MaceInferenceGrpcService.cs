using System.Diagnostics;
using Grpc.Core;
using MACE.Core;
using MACE.Protos;
using Microsoft.Extensions.Options;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Prometheus;

namespace MACE.Services;

public sealed class MaceInferenceGrpcService : MaceInference.MaceInferenceBase
{
    // Static: shared across all per-request instances so uptime is from app start, not request start.
    private static readonly Stopwatch _uptime = Stopwatch.StartNew();

    // Prometheus metrics — static so they are registered once with the global registry.
    private static readonly Histogram InferDuration = Metrics.CreateHistogram(
        "mace_infer_duration_seconds",
        "VMP inference wall-clock time.",
        new HistogramConfiguration
        {
            Buckets = new[] { 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0 }
        });

    private static readonly Gauge PoolAvailable = Metrics.CreateGauge(
        "mace_pool_available",
        "Number of idle inference pool slots.");

    private static readonly Counter InferRequests = Metrics.CreateCounter(
        "mace_infer_requests_total",
        "Total Infer RPC calls by outcome.",
        new CounterConfiguration { LabelNames = new[] { "status" } });

    private static readonly Counter UpdatePriorsRequests = Metrics.CreateCounter(
        "mace_update_priors_requests_total",
        "Total UpdatePriors RPC calls by verdict.",
        new CounterConfiguration { LabelNames = new[] { "verdict" } });

    private readonly InferencePool       _pool;
    private readonly PriorUpdateService  _priorUpdate;
    private readonly InferenceOptions    _opts;
    private readonly ILogger<MaceInferenceGrpcService> _logger;

    public MaceInferenceGrpcService(
        InferencePool pool,
        PriorUpdateService priorUpdate,
        IOptions<InferenceOptions> opts,
        ILogger<MaceInferenceGrpcService> logger)
    {
        _pool        = pool;
        _priorUpdate = priorUpdate;
        _opts        = opts.Value;
        _logger      = logger;
    }

    // -------------------------------------------------------------------------
    // Infer
    // -------------------------------------------------------------------------

    public override async Task<InferResponse> Infer(
        InferRequest request, ServerCallContext context)
    {
        ValidateInferRequest(request);

        var priors = BuildModelData(request);

        Discrete? warmStart = null;
        if (request.WarmStart.Count > 0)
        {
            var vec = Vector.FromArray(request.WarmStart.ToArray());
            warmStart = new Discrete(vec);
        }

        int numObs = request.Annotations.Count(a => a != -1);
        if (numObs < _opts.MinSensorsForInference)
        {
            _logger.LogDebug(
                "Incident {Id}: only {N} observations (< min {Min}); using raw-max fallback.",
                request.IncidentId, numObs, _opts.MinSensorsForInference);
            InferRequests.WithLabels("fallback").Inc();
            return BuildFallbackResponse(request, numObs);
        }

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(context.CancellationToken);
        cts.CancelAfter(_opts.PoolAcquireTimeoutMs);

        PooledInference lease;
        try
        {
            lease = await _pool.AcquireAsync(cts.Token);
        }
        catch (OperationCanceledException) when (!context.CancellationToken.IsCancellationRequested)
        {
            _logger.LogWarning("Pool exhausted while serving incident {Id}.", request.IncidentId);
            InferRequests.WithLabels("timeout").Inc();
            throw new RpcException(new Status(StatusCode.Unavailable,
                "Inference pool exhausted — retry after a moment."));
        }

        OnlineInferenceResult result;
        long elapsedMs;
        using (lease)
        using (InferDuration.NewTimer())
        {
            var sw = Stopwatch.StartNew();
            result    = lease.Inferencer.InferOnline(request.Annotations.ToArray(), priors, warmStart);
            elapsedMs = sw.ElapsedMilliseconds;
        }

        PoolAvailable.Set(_pool.Available);
        InferRequests.WithLabels("success").Inc();

        _logger.LogDebug(
            "Incident {Id}: threat={Level} conf={Conf:F3} in {Ms}ms.",
            request.IncidentId, result.ThreatLevel, result.Confidence, elapsedMs);

        return BuildInferResponse(request, result, numObs, elapsedMs);
    }

    // -------------------------------------------------------------------------
    // UpdatePriors
    // -------------------------------------------------------------------------

    public override Task<UpdatePriorsResponse> UpdatePriors(
        UpdatePriorsRequest request, ServerCallContext context)
    {
        Verdict verdict;
        try
        {
            verdict = PriorUpdateService.ParseVerdict(request.Verdict);
        }
        catch (ArgumentException ex)
        {
            throw new RpcException(new Status(StatusCode.InvalidArgument, ex.Message));
        }

        double lr = request.LearningRate > 0 ? request.LearningRate : 0.5;

        var response = new UpdatePriorsResponse();
        foreach (var sensor in request.Sensors)
        {
            var current = new BetaParameters(sensor.CurrentTheta.Alpha, sensor.CurrentTheta.Beta);
            var updated = _priorUpdate.UpdateTheta(
                current, sensor.SpammerProbMean, sensor.Annotation, verdict, lr);

            response.UpdatedThetas.Add(new UpdatedTheta
            {
                SensorTypeIndex = sensor.SensorTypeIndex,
                Alpha           = updated.Alpha,
                Beta            = updated.Beta
            });
        }

        UpdatePriorsRequests.WithLabels(request.Verdict.ToLowerInvariant()).Inc();
        return Task.FromResult(response);
    }

    // -------------------------------------------------------------------------
    // Health
    // -------------------------------------------------------------------------

    public override Task<HealthResponse> Health(
        HealthRequest request, ServerCallContext context)
    {
        PoolAvailable.Set(_pool.Available);
        return Task.FromResult(new HealthResponse
        {
            Status        = "healthy",
            PoolAvailable = _pool.Available,
            PoolTotal     = _pool.Total,
            UptimeSeconds = (long)_uptime.Elapsed.TotalSeconds
        });
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private void ValidateInferRequest(InferRequest req)
    {
        if (req.Annotations.Count != _opts.NumSensorTypes)
            throw new RpcException(new Status(StatusCode.InvalidArgument,
                $"annotations must have {_opts.NumSensorTypes} elements, got {req.Annotations.Count}."));

        if (req.ThetaPriors.Count != _opts.NumSensorTypes)
            throw new RpcException(new Status(StatusCode.InvalidArgument,
                $"theta_priors must have {_opts.NumSensorTypes} elements, got {req.ThetaPriors.Count}."));

        if (req.PhiPriors.Count != _opts.NumSensorTypes)
            throw new RpcException(new Status(StatusCode.InvalidArgument,
                $"phi_priors must have {_opts.NumSensorTypes} elements, got {req.PhiPriors.Count}."));

        foreach (var theta in req.ThetaPriors)
        {
            if (theta.Alpha <= 0 || theta.Beta <= 0)
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    $"Beta parameters must be positive; got alpha={theta.Alpha}, beta={theta.Beta}."));
        }

        foreach (var phi in req.PhiPriors)
        {
            if (phi.Pseudocounts.Count != _opts.NumCategories)
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    $"Each phi prior must have {_opts.NumCategories} pseudocounts."));

            if (phi.Pseudocounts.Any(c => c <= 0))
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    "Dirichlet pseudocounts must all be positive."));
        }

        if (req.WarmStart.Count > 0 && req.WarmStart.Count != _opts.NumCategories)
            throw new RpcException(new Status(StatusCode.InvalidArgument,
                $"warm_start must be empty or have {_opts.NumCategories} elements."));
    }

    private ModelData BuildModelData(InferRequest req)
    {
        return new ModelData
        {
            ThetaDist = req.ThetaPriors
                .Select(t => new Beta(t.Alpha, t.Beta))
                .ToArray(),
            PhiDist = req.PhiPriors
                .Select(p => new Dirichlet(p.Pseudocounts.ToArray()))
                .ToArray()
        };
    }

    private InferResponse BuildInferResponse(
        InferRequest req, OnlineInferenceResult result, int numObs, long ms)
    {
        var resp = new InferResponse
        {
            IncidentId      = req.IncidentId,
            ThreatLevel     = result.ThreatLevel,
            Confidence      = result.Confidence,
            Entropy         = result.Entropy,
            NumObservations = numObs,
            InferenceMs     = ms
        };

        resp.TDist.AddRange(result.TDist.GetProbs().ToArray());

        for (int j = 0; j < req.Annotations.Count; j++)
        {
            if (req.Annotations[j] == -1) continue;
            double sp = result.SDist[j].GetProbTrue();
            resp.SensorReliability.Add(new SensorReliability
            {
                SensorTypeIndex = j,
                Annotation      = req.Annotations[j],
                SpammerProb     = sp,
                Reliability     = 1.0 - sp
            });
        }

        return resp;
    }

    // Fallback when too few sensors are present to run MACE.
    // Returns the max observed annotation as threat_level with a uniform (maximum-entropy) distribution.
    private InferResponse BuildFallbackResponse(InferRequest req, int numObs)
    {
        int maxAnnotation = req.Annotations.Where(a => a >= 0).DefaultIfEmpty(0).Max();
        double uniform    = 1.0 / _opts.NumCategories;
        double maxEntropy = Math.Log(_opts.NumCategories);

        var resp = new InferResponse
        {
            IncidentId      = req.IncidentId,
            ThreatLevel     = maxAnnotation,
            Confidence      = uniform,
            Entropy         = maxEntropy,
            NumObservations = numObs,
            InferenceMs     = 0
        };
        resp.TDist.AddRange(Enumerable.Repeat(uniform, _opts.NumCategories));
        return resp;
    }
}
