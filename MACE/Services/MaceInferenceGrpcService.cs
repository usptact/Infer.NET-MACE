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

        int numObs = request.SensorReadings.Count(a => a != -1);

        // DEBUG: log full request contents before any processing
        if (_logger.IsEnabled(LogLevel.Debug))
            _logger.LogDebug("Infer request:\n{Detail}",
                FormatInferRequest(request, numObs));

        // ── Insufficient observations → raw-max fallback ──────────────────────
        if (numObs < _opts.MinSensorsForInference)
        {
            int maxAnn = request.SensorReadings.Where(a => a >= 0).DefaultIfEmpty(0).Max();
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
                    request.SensorReadings.ToArray(), priors, warmStart);
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

        // ── Resolve the gold true level and validate per-sensor payload ────────
        // A FALSE_ALARM pins the gold level to CLEAR (0); a TRUE_ALARM uses the
        // operator-resolved level supplied on the request.
        int goldLevel = verdict == Verdict.FalseAlarm ? 0 : request.TrueThreatLevel;
        ValidateUpdatePriorsRequest(request, goldLevel);

        double lr = request.LearningRate > 0 ? request.LearningRate : 0.5;

        // ── DEBUG: log incoming request before computing ───────────────────────
        if (_logger.IsEnabled(LogLevel.Debug))
            _logger.LogDebug("UpdatePriors request:\n{Detail}",
                FormatUpdatePriorsRequest(request, lr, goldLevel));

        // ── Compute updated priors (single-incident EM step) ───────────────────
        var response = new UpdatePriorsResponse();
        foreach (var sensor in request.Sensors)
        {
            var currentTheta = new BetaParameters(sensor.CurrentTheta.Alpha, sensor.CurrentTheta.Beta);
            var currentPhi   = sensor.CurrentPhi.Pseudocounts.ToArray();
            var updated = _priorUpdate.UpdateBeliefs(
                currentTheta, currentPhi, sensor.SensorReading, goldLevel, lr);

            response.UpdatedThetas.Add(new UpdatedTheta
            {
                SensorTypeIndex = sensor.SensorTypeIndex,
                Alpha           = updated.Theta.Alpha,
                Beta            = updated.Theta.Beta
            });

            var updatedPhi = new UpdatedPhi { SensorTypeIndex = sensor.SensorTypeIndex };
            updatedPhi.Pseudocounts.AddRange(updated.Phi);
            response.UpdatedPhis.Add(updatedPhi);
        }

        // ── INFO: one compact line ─────────────────────────────────────────────
        _logger.LogInformation(
            "UpdatePriors  verdict={Verdict}  gold={Gold}  sensors={Count}  lr={Lr:F2}",
            request.Verdict, LevelName(goldLevel), request.Sensors.Count, lr);

        // ── DEBUG: before/after for every sensor ──────────────────────────────
        if (_logger.IsEnabled(LogLevel.Debug))
            _logger.LogDebug("UpdatePriors result:\n{Detail}",
                FormatUpdatePriorsResult(request, response, lr, goldLevel));

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
        if (req.SensorReadings.Count != _opts.NumSensorTypes)
            throw new RpcException(new Status(StatusCode.InvalidArgument,
                $"sensor_readings must have {_opts.NumSensorTypes} elements, got {req.SensorReadings.Count}."));

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

    private void ValidateUpdatePriorsRequest(UpdatePriorsRequest req, int goldLevel)
    {
        if (goldLevel < 0 || goldLevel >= _opts.NumThreatLevels)
            throw new RpcException(new Status(StatusCode.InvalidArgument,
                $"true_threat_level must be in [0, {_opts.NumThreatLevels}), got {goldLevel}."));

        foreach (var sensor in req.Sensors)
        {
            if (sensor.SensorReading < -1 || sensor.SensorReading >= _opts.NumThreatLevels)
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    $"sensor_reading must be -1 or in [0, {_opts.NumThreatLevels}), " +
                    $"got {sensor.SensorReading} for sensor {sensor.SensorTypeIndex}."));

            if (sensor.CurrentTheta is null || sensor.CurrentTheta.Alpha <= 0 || sensor.CurrentTheta.Beta <= 0)
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    $"current_theta must have positive alpha and beta for sensor {sensor.SensorTypeIndex}."));

            // φ is required by the EM update: an absent sensor still needs a
            // well-formed prior so the returned pseudocounts round-trip unchanged.
            if (sensor.CurrentPhi is null || sensor.CurrentPhi.Pseudocounts.Count != _opts.NumThreatLevels)
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    $"current_phi must have {_opts.NumThreatLevels} pseudocounts for sensor {sensor.SensorTypeIndex}, " +
                    $"got {sensor.CurrentPhi?.Pseudocounts.Count ?? 0}."));

            if (sensor.CurrentPhi.Pseudocounts.Any(c => c <= 0))
                throw new RpcException(new Status(StatusCode.InvalidArgument,
                    $"current_phi pseudocounts must all be positive for sensor {sensor.SensorTypeIndex}."));
        }
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
        resp.ThreatDist.AddRange(result.ThreatDist.GetProbs().ToArray());
        for (int j = 0; j < req.SensorReadings.Count; j++)
        {
            if (req.SensorReadings[j] == -1) continue;
            double sp = result.FaultDist[j].GetProbTrue();
            resp.SensorReliability.Add(new SensorReliability
            {
                SensorTypeIndex = j,
                SensorReading   = req.SensorReadings[j],
                FaultProb       = sp,
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
        resp.ThreatDist.AddRange(Enumerable.Repeat(uniform, _opts.NumThreatLevels));
        return resp;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Debug formatters — only called when LogLevel.Debug is enabled
    // ─────────────────────────────────────────────────────────────────────────

    private string FormatInferRequest(InferRequest req, int numObs)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"  incident_id  : {req.IncidentId}");
        sb.AppendLine($"  sensor_readings : [{string.Join(", ", req.SensorReadings)}]  ({numObs} of {req.SensorReadings.Count} sensors present)");

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
        for (int i = 0; i < resp.ThreatDist.Count; i++)
        {
            string label  = i < ThreatLevelNames.Length ? ThreatLevelNames[i] : $"L{i}";
            string marker = i == resp.ThreatLevel ? " ← argmax" : string.Empty;
            sb.AppendLine($"    {label,-8} = {resp.ThreatDist[i]:F4}{marker}");
        }

        if (resp.SensorReliability.Count > 0)
        {
            sb.AppendLine("  sensors :");
            foreach (var sr in resp.SensorReliability)
                sb.AppendLine(
                    $"    [{sr.SensorTypeIndex}]  reading={sr.SensorReading}  " +
                    $"fault={sr.FaultProb:F3}  reliable={sr.Reliability:F3}");
        }

        sb.Append($"  elapsed : {resp.InferenceMs}ms");
        return sb.ToString();
    }

    private static string FormatUpdatePriorsRequest(UpdatePriorsRequest req, double lr, int goldLevel)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"  verdict          : {req.Verdict}");
        sb.AppendLine($"  gold_true_level  : {goldLevel}");
        sb.AppendLine($"  learning_rate    : {lr:F2}");
        sb.Append($"  sensors          : {req.Sensors.Count}");
        return sb.ToString();
    }

    private string FormatUpdatePriorsResult(
        UpdatePriorsRequest req, UpdatePriorsResponse resp, double lr, int goldLevel)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"  verdict={req.Verdict}  gold={goldLevel}  lr={lr:F2}");

        foreach (var upd in resp.UpdatedThetas)
        {
            var src = req.Sensors.FirstOrDefault(s => s.SensorTypeIndex == upd.SensorTypeIndex);
            var phi = resp.UpdatedPhis.FirstOrDefault(p => p.SensorTypeIndex == upd.SensorTypeIndex);

            string status;
            if (src is null)
            {
                status = "unknown";
            }
            else if (src.SensorReading == -1)
            {
                status = "absent — no change";
            }
            else if (src.SensorReading == goldLevel)
            {
                status = "matched gold, mostly reliable";
            }
            else
            {
                status = "disagreed with gold, penalised";
            }

            double origAlpha = src?.CurrentTheta.Alpha ?? upd.Alpha;
            double origBeta  = src?.CurrentTheta.Beta  ?? upd.Beta;

            bool changed = Math.Abs(upd.Alpha - origAlpha) > 1e-9 || Math.Abs(upd.Beta - origBeta) > 1e-9;
            string delta = changed
                ? $"α {origAlpha:F3}→{upd.Alpha:F3}  β {origBeta:F3}→{upd.Beta:F3}"
                : $"α {upd.Alpha:F3}  β {upd.Beta:F3}  (unchanged)";

            string phiStr = phi is not null
                ? $"  φ=[{string.Join(",", phi.Pseudocounts.Select(c => c.ToString("F3")))}]"
                : string.Empty;

            sb.AppendLine($"  sensor[{upd.SensorTypeIndex}]  {delta}{phiStr}   {status}");
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
