using System.Diagnostics;
using System.Text;
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
    // ── Prometheus metrics ────────────────────────────────────────────────────
    // Static: registered once with the global registry regardless of how many
    // per-request instances the gRPC framework creates.

    private static readonly Histogram InferDuration = Metrics.CreateHistogram(
        "mace_infer_duration_seconds",
        "VMP inference wall-clock time.",
        new HistogramConfiguration
        {
            Buckets = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        });

    private static readonly Gauge PoolAvailable = Metrics.CreateGauge(
        "mace_pool_available", "Idle inference pool slots.");

    private static readonly Counter InferRequests = Metrics.CreateCounter(
        "mace_infer_requests_total", "Total Infer RPC calls.",
        new CounterConfiguration { LabelNames = ["status"] });

    private static readonly Counter UpdatePriorsRequests = Metrics.CreateCounter(
        "mace_update_priors_requests_total", "Total UpdatePriors RPC calls.",
        new CounterConfiguration { LabelNames = ["verdict"] });

    // Static stopwatch so uptime is measured from class load (≈ app start),
    // not from request arrival.  The gRPC framework instantiates this class
    // per-request, so instance fields would reset on every call.
    private static readonly Stopwatch _uptime = Stopwatch.StartNew();

    private static readonly string[] ThreatLevelNames =
        ["CLEAR", "LOW", "MEDIUM", "HIGH", "CRITICAL"];

    // ── Instance state ────────────────────────────────────────────────────────

    private readonly IInferencePool     _pool;
    private readonly PriorUpdateService _priorUpdate;
    private readonly InferenceOptions   _opts;
    private readonly ILogger<MaceInferenceGrpcService> _logger;

    public MaceInferenceGrpcService(
        IInferencePool pool,
        PriorUpdateService priorUpdate,
        IOptions<InferenceOptions> opts,
        ILogger<MaceInferenceGrpcService> logger)
    {
        _pool        = pool;
        _priorUpdate = priorUpdate;
        _opts        = opts.Value;
        _logger      = logger;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Infer
    // ─────────────────────────────────────────────────────────────────────────

    public override async Task<InferResponse> Infer(
        InferRequest request, ServerCallContext context)
    {
        ValidateInferRequest(request);

        int numObs = request.Annotations.Count(a => a != -1);

        // DEBUG: log full request contents before any processing
        if (_logger.IsEnabled(LogLevel.Debug))
            _logger.LogDebug("Infer request:\n{Detail}",
                FormatInferRequest(request, numObs));

        // ── Insufficient observations → raw-max fallback ──────────────────────
        if (numObs < _opts.MinSensorsForInference)
        {
            int maxAnn = request.Annotations.Where(a => a >= 0).DefaultIfEmpty(0).Max();
            _logger.LogWarning(
                "Infer  incident={Id}  obs={Obs}/{Total} < min={Min}  →  fallback  " +
                "max-ann={MaxAnn}  level={Level}",
                request.IncidentId, numObs, _opts.NumSensorTypes,
                _opts.MinSensorsForInference, maxAnn, LevelName(maxAnn));

            InferRequests.WithLabels("fallback").Inc();
            return BuildFallbackResponse(request, numObs, maxAnn);
        }

        // ── Build priors and optional warm-start ──────────────────────────────
        var priors    = BuildModelData(request);
        Discrete? warmStart = null;
        if (request.WarmStart.Count > 0)
        {
            warmStart = new Discrete(Vector.FromArray(request.WarmStart.ToArray()));
            _logger.LogDebug(
                "Infer  incident={Id}  warm-start provided ({Cats} categories)",
                request.IncidentId, request.WarmStart.Count);
        }

        // ── Acquire pool slot ─────────────────────────────────────────────────
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(context.CancellationToken);
        cts.CancelAfter(_opts.PoolAcquireTimeoutMs);

        PooledInference lease;
        try
        {
            lease = await _pool.AcquireAsync(cts.Token);
            _logger.LogDebug(
                "Infer  incident={Id}  pool slot acquired  ({Available}/{Total} remaining)",
                request.IncidentId, _pool.Available, _pool.Total);
        }
        catch (OperationCanceledException) when (!context.CancellationToken.IsCancellationRequested)
        {
            _logger.LogWarning(
                "Infer  incident={Id}  pool exhausted after {Timeout}ms  →  UNAVAILABLE",
                request.IncidentId, _opts.PoolAcquireTimeoutMs);
            InferRequests.WithLabels("timeout").Inc();
            throw new RpcException(new Status(StatusCode.Unavailable,
                "Inference pool exhausted — retry after a moment."));
        }

        // ── Run VMP inference ─────────────────────────────────────────────────
        OnlineInferenceResult result;
        long elapsedMs;

        try
        {
            using (lease)
            using (InferDuration.NewTimer())
            {
                var sw = Stopwatch.StartNew();
                result    = lease.Inferencer.InferOnline(
                    request.Annotations.ToArray(), priors, warmStart);
                elapsedMs = sw.ElapsedMilliseconds;
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex,
                "Infer  incident={Id}  inference threw an exception  →  INTERNAL",
                request.IncidentId);
            InferRequests.WithLabels("error").Inc();
            throw new RpcException(new Status(StatusCode.Internal,
                "Inference failed; see service logs for details."));
        }

        PoolAvailable.Set(_pool.Available);
        InferRequests.WithLabels("success").Inc();

        // ── INFO: one compact line per call ───────────────────────────────────
        _logger.LogInformation(
            "Infer  incident={Id}  obs={Obs}/{Total}  →  {LevelName}({Level})  " +
            "conf={Conf:F3}  entropy={Entropy:F3}  {Ms}ms",
            request.IncidentId, numObs, _opts.NumSensorTypes,
            LevelName(result.ThreatLevel), result.ThreatLevel,
            result.Confidence, result.Entropy, elapsedMs);

        // ── DEBUG: full posterior detail ──────────────────────────────────────
        var response = BuildInferResponse(request, result, numObs, elapsedMs);
        if (_logger.IsEnabled(LogLevel.Debug))
            _logger.LogDebug("Infer result:\n{Detail}", FormatInferResult(response));

        return response;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UpdatePriors
    // ─────────────────────────────────────────────────────────────────────────

    public override Task<UpdatePriorsResponse> UpdatePriors(
        UpdatePriorsRequest request, ServerCallContext context)
    {
        // ── Parse verdict ─────────────────────────────────────────────────────
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

        // ── DEBUG: log incoming request before computing ───────────────────────
        if (_logger.IsEnabled(LogLevel.Debug))
            _logger.LogDebug("UpdatePriors request:\n{Detail}",
                FormatUpdatePriorsRequest(request, lr));

        // ── Compute updated priors ─────────────────────────────────────────────
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

        // ── INFO: one compact line ─────────────────────────────────────────────
        _logger.LogInformation(
            "UpdatePriors  verdict={Verdict}  sensors={Count}  lr={Lr:F2}",
            request.Verdict, request.Sensors.Count, lr);

        // ── DEBUG: before/after for every sensor ──────────────────────────────
        if (_logger.IsEnabled(LogLevel.Debug))
            _logger.LogDebug("UpdatePriors result:\n{Detail}",
                FormatUpdatePriorsResult(request, response, lr));

        UpdatePriorsRequests.WithLabels(request.Verdict.ToLowerInvariant()).Inc();
        return Task.FromResult(response);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Health
    // ─────────────────────────────────────────────────────────────────────────

    public override Task<HealthResponse> Health(
        HealthRequest request, ServerCallContext context)
    {
        PoolAvailable.Set(_pool.Available);

        // Health is called frequently by Kubernetes; log only at DEBUG to avoid
        // flooding the terminal with probe traffic.
        _logger.LogDebug(
            "Health  pool={Available}/{Total}  uptime={Uptime}s",
            _pool.Available, _pool.Total, (long)_uptime.Elapsed.TotalSeconds);

        return Task.FromResult(new HealthResponse
        {
            Status        = "healthy",
            PoolAvailable = _pool.Available,
            PoolTotal     = _pool.Total,
            UptimeSeconds = (long)_uptime.Elapsed.TotalSeconds
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Validation
    // ─────────────────────────────────────────────────────────────────────────

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
            if (phi.Pseudocounts.Count != _opts.NumThreatLevels)
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    $"Each phi prior needs {_opts.NumThreatLevels} pseudocounts, got {phi.Pseudocounts.Count}."));

            if (phi.Pseudocounts.Any(c => c <= 0))
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    "Dirichlet pseudocounts must all be positive."));
        }

        if (req.WarmStart.Count > 0 && req.WarmStart.Count != _opts.NumThreatLevels)
            throw new RpcException(new Status(StatusCode.InvalidArgument,
                $"warm_start must be empty or have {_opts.NumThreatLevels} elements, got {req.WarmStart.Count}."));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Response builders
    // ─────────────────────────────────────────────────────────────────────────

    private ModelData BuildModelData(InferRequest req) =>
        new()
        {
            ThetaDist = req.ThetaPriors.Select(t => new Beta(t.Alpha, t.Beta)).ToArray(),
            PhiDist   = req.PhiPriors.Select(p => new Dirichlet(p.Pseudocounts.ToArray())).ToArray()
        };

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
        resp.TDist.AddRange(result.ThreatDist.GetProbs().ToArray());
        for (int j = 0; j < req.Annotations.Count; j++)
        {
            if (req.Annotations[j] == -1) continue;
            double sp = result.FaultDist[j].GetProbTrue();
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

    private InferResponse BuildFallbackResponse(InferRequest req, int numObs, int maxAnnotation)
    {
        double uniform    = 1.0 / _opts.NumThreatLevels;
        double maxEntropy = Math.Log(_opts.NumThreatLevels);
        var resp = new InferResponse
        {
            IncidentId      = req.IncidentId,
            ThreatLevel     = maxAnnotation,
            Confidence      = uniform,
            Entropy         = maxEntropy,
            NumObservations = numObs,
            InferenceMs     = 0
        };
        resp.TDist.AddRange(Enumerable.Repeat(uniform, _opts.NumThreatLevels));
        return resp;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Debug formatters — only called when LogLevel.Debug is enabled
    // ─────────────────────────────────────────────────────────────────────────

    private string FormatInferRequest(InferRequest req, int numObs)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"  incident_id  : {req.IncidentId}");
        sb.AppendLine($"  annotations  : [{string.Join(", ", req.Annotations)}]  ({numObs} of {req.Annotations.Count} sensors present)");

        var thetaParts = req.ThetaPriors.Select(t => $"α={t.Alpha:F1}/β={t.Beta:F1}");
        sb.AppendLine($"  theta_priors : [{string.Join("  ", thetaParts)}]");

        bool allSamePhi = req.PhiPriors.All(p =>
            p.Pseudocounts.Count == req.PhiPriors[0].Pseudocounts.Count &&
            p.Pseudocounts.SequenceEqual(req.PhiPriors[0].Pseudocounts));

        if (allSamePhi && req.PhiPriors.Count > 0)
        {
            var first = $"[{string.Join(",", req.PhiPriors[0].Pseudocounts.Select(c => c.ToString("F0")))}]";
            sb.AppendLine($"  phi_priors   : all identical {first}");
        }
        else
        {
            var phiParts = req.PhiPriors.Select(p =>
                $"[{string.Join(",", p.Pseudocounts.Select(c => c.ToString("F0")))}]");
            sb.AppendLine($"  phi_priors   : [{string.Join("  ", phiParts)}]");
        }

        if (req.WarmStart.Count > 0)
        {
            var ws = string.Join(", ", req.WarmStart.Select(w => $"{w:F4}"));
            sb.Append($"  warm_start   : [{ws}]");
        }
        else
        {
            sb.Append("  warm_start   : none");
        }

        return sb.ToString();
    }

    private string FormatInferResult(InferResponse resp)
    {
        var sb = new StringBuilder();
        sb.AppendLine("  t_dist :");
        for (int i = 0; i < resp.TDist.Count; i++)
        {
            string label  = i < ThreatLevelNames.Length ? ThreatLevelNames[i] : $"L{i}";
            string marker = i == resp.ThreatLevel ? " ← argmax" : string.Empty;
            sb.AppendLine($"    {label,-8} = {resp.TDist[i]:F4}{marker}");
        }

        if (resp.SensorReliability.Count > 0)
        {
            sb.AppendLine("  sensors :");
            foreach (var sr in resp.SensorReliability)
                sb.AppendLine(
                    $"    [{sr.SensorTypeIndex}]  ann={sr.Annotation}  " +
                    $"spammer={sr.SpammerProb:F3}  reliable={sr.Reliability:F3}");
        }

        sb.Append($"  elapsed : {resp.InferenceMs}ms");
        return sb.ToString();
    }

    private static string FormatUpdatePriorsRequest(UpdatePriorsRequest req, double lr)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"  verdict       : {req.Verdict}");
        sb.AppendLine($"  learning_rate : {lr:F2}");
        sb.Append($"  sensors       : {req.Sensors.Count}");
        return sb.ToString();
    }

    private static string FormatUpdatePriorsResult(
        UpdatePriorsRequest req, UpdatePriorsResponse resp, double lr)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"  verdict={req.Verdict}  lr={lr:F2}");

        foreach (var upd in resp.UpdatedThetas)
        {
            var src = req.Sensors.FirstOrDefault(s => s.SensorTypeIndex == upd.SensorTypeIndex);

            string status;
            if (src is null)
            {
                status = "unknown";
            }
            else if (src.Annotation == -1)
            {
                status = "absent — no change";
            }
            else if (src.Annotation >= 2)
            {
                status = req.Verdict == "TRUE_ALARM" ? "flagged correctly, reinforced" : "false alarm contributor";
            }
            else
            {
                status = req.Verdict == "TRUE_ALARM" ? "missed threat, penalised" : "correctly quiet, reinforced";
            }

            double origAlpha = src?.CurrentTheta.Alpha ?? upd.Alpha;
            double origBeta  = src?.CurrentTheta.Beta  ?? upd.Beta;

            bool changed = Math.Abs(upd.Alpha - origAlpha) > 1e-9 || Math.Abs(upd.Beta - origBeta) > 1e-9;
            string delta = changed
                ? $"α {origAlpha:F3}→{upd.Alpha:F3}  β {origBeta:F3}→{upd.Beta:F3}"
                : $"α {upd.Alpha:F3}  β {upd.Beta:F3}  (unchanged)";

            sb.AppendLine($"  sensor[{upd.SensorTypeIndex}]  {delta}   {status}");
        }

        // Trim trailing newline
        return sb.ToString().TrimEnd();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utilities
    // ─────────────────────────────────────────────────────────────────────────

    private static string LevelName(int level) =>
        level >= 0 && level < ThreatLevelNames.Length ? ThreatLevelNames[level] : $"L{level}";
}
