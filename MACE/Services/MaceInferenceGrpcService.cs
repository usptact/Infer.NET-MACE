using System.Diagnostics;
using Grpc.Core;
using MACE.Core;
using MACE.Protos;
using Microsoft.Extensions.Options;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace MACE.Services;

public sealed class MaceInferenceGrpcService : MaceInference.MaceInferenceBase
{
    private readonly InferencePool       _pool;
    private readonly PriorUpdateService  _priorUpdate;
    private readonly InferenceOptions    _opts;
    private readonly ILogger<MaceInferenceGrpcService> _logger;
    private readonly Stopwatch           _uptime = Stopwatch.StartNew();

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
                "Incident {Id}: only {N} observations, below threshold {Min}; using raw-max fallback.",
                request.IncidentId, numObs, _opts.MinSensorsForInference);
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
            throw new RpcException(new Status(StatusCode.Unavailable,
                "Inference pool exhausted — retry after a moment."));
        }

        var sw = Stopwatch.StartNew();
        OnlineInferenceResult result;
        using (lease)
        {
            result = lease.Inferencer.InferOnline(
                request.Annotations.ToArray(),
                priors,
                warmStart);
        }
        sw.Stop();

        _logger.LogDebug(
            "Incident {Id}: threat={Level} conf={Conf:F3} in {Ms}ms.",
            request.IncidentId, result.ThreatLevel, result.Confidence, sw.ElapsedMilliseconds);

        return BuildInferResponse(request, result, numObs, sw.ElapsedMilliseconds);
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
            var current = new BetaParameters(
                sensor.CurrentTheta.Alpha,
                sensor.CurrentTheta.Beta);

            var updated = _priorUpdate.UpdateTheta(
                current,
                sensor.SpammerProbMean,
                sensor.Annotation,
                verdict,
                lr);

            response.UpdatedThetas.Add(new UpdatedTheta
            {
                SensorTypeIndex = sensor.SensorTypeIndex,
                Alpha           = updated.Alpha,
                Beta            = updated.Beta
            });
        }

        return Task.FromResult(response);
    }

    // -------------------------------------------------------------------------
    // Health
    // -------------------------------------------------------------------------

    public override Task<HealthResponse> Health(
        HealthRequest request, ServerCallContext context)
    {
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

        foreach (var phi in req.PhiPriors)
        {
            if (phi.Pseudocounts.Count != _opts.NumCategories)
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    $"Each phi prior must have {_opts.NumCategories} pseudocounts."));
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
            IncidentId     = req.IncidentId,
            ThreatLevel    = result.ThreatLevel,
            Confidence     = result.Confidence,
            Entropy        = result.Entropy,
            NumObservations = numObs,
            InferenceMs    = ms
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

    // Fallback when too few sensors observed the incident to run MACE.
    // Returns the highest observed annotation as threat_level with low confidence.
    private InferResponse BuildFallbackResponse(InferRequest req, int numObs)
    {
        int maxAnnotation = req.Annotations.Where(a => a >= 0).DefaultIfEmpty(0).Max();
        var uniform = Enumerable.Repeat(1.0 / _opts.NumCategories, _opts.NumCategories).ToArray();

        var resp = new InferResponse
        {
            IncidentId      = req.IncidentId,
            ThreatLevel     = maxAnnotation,
            Confidence      = 1.0 / _opts.NumCategories,
            Entropy         = Math.Log(_opts.NumCategories),
            NumObservations = numObs,
            InferenceMs     = 0
        };
        resp.TDist.AddRange(uniform);
        return resp;
    }
}
