using MACE.Core;
using MACE.Services;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.Options;
using Prometheus;
using Serilog;
using Serilog.Core;
using Serilog.Events;

// ── Phase 1: bootstrap logger (used while DI container is being built) ────
Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Debug()
    .Enrich.With<ShortClassNameEnricher>()
    .WriteTo.Console(outputTemplate:
        "[{Timestamp:yyyy-MM-dd HH:mm:ss.fff} {Level:u3}] [{ShortContext,-30}] {Message:lj}{NewLine}{Exception}")
    .CreateBootstrapLogger();

var startLog = Log.ForContext("SourceContext", "Program");

try
{
    startLog.Information("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    startLog.Information("MACE Inference Service starting");

    var builder = WebApplication.CreateBuilder(args);
    builder.Configuration.AddEnvironmentVariables();

    // ── Phase 2: full Serilog config (MinimumLevel read from appsettings) ─────
    // WriteTo sink is defined here in code so the ShortContext enricher and
    // output template are always applied regardless of which appsettings file
    // is active.  MinimumLevel overrides live in appsettings.json /
    // appsettings.Development.json under the "Serilog" key.
    builder.Services.AddSerilog(loggerConfig => loggerConfig
        .ReadFrom.Configuration(builder.Configuration)
        .Enrich.FromLogContext()
        .Enrich.With<ShortClassNameEnricher>()
        .WriteTo.Console(outputTemplate:
            "[{Timestamp:yyyy-MM-dd HH:mm:ss.fff} {Level:u3}] [{ShortContext,-30}] {Message:lj}{NewLine}{Exception}"));

    // ── DI registrations ──────────────────────────────────────────────────────
    builder.Services.Configure<InferenceOptions>(builder.Configuration.GetSection("Inference"));
    builder.Services.AddSingleton<IInferencePool, InferencePool>();
    builder.Services.AddSingleton<PriorUpdateService>();
    builder.Services.AddGrpc();

    // gRPC requires HTTP/2.  TLS is terminated at the load-balancer level in
    // the Kubernetes deployment, so we listen on cleartext HTTP/2 (H2C).
    // Port 9090 is HTTP/1.1 only — Prometheus scrapers do not support HTTP/2.
    builder.WebHost.ConfigureKestrel(o =>
    {
        o.ListenAnyIP(8080, lo => lo.Protocols = HttpProtocols.Http2);
        o.ListenAnyIP(9090, lo => lo.Protocols = HttpProtocols.Http1);
    });

    var app = builder.Build();

    // Log resolved configuration before pool warmup so the values are visible
    // in the output before the potentially slow CreateModel() calls.
    var opts = app.Services.GetRequiredService<IOptions<InferenceOptions>>().Value;
    startLog.Information(
        "Config  sensors={Sensors}  categories={Categories}  pool={Pool}  " +
        "min-obs={MinObs}  pool-timeout={TimeoutMs}ms",
        opts.NumSensorTypes, opts.NumCategories, opts.PoolSize,
        opts.MinSensorsForInference, opts.PoolAcquireTimeoutMs);

    // Force the singleton to construct now so the pool is fully warm before
    // the first request arrives (otherwise the first caller pays the cost).
    _ = app.Services.GetRequiredService<IInferencePool>();

    app.MapGrpcService<MaceInferenceGrpcService>();

    // Prometheus metrics on the HTTP/1.1 port only.
    app.UseHttpMetrics();
    app.MapMetrics("/metrics").RequireHost("*:9090");

    app.MapGet("/", () => "MACE Inference Service — use a gRPC client.");

    // ── Lifecycle hooks ───────────────────────────────────────────────────────
    var lifetime = app.Services.GetRequiredService<IHostApplicationLifetime>();

    lifetime.ApplicationStarted.Register(() =>
    {
        startLog.Information("gRPC    listening on :8080 (H2C)");
        startLog.Information("Metrics listening on :9090");
        startLog.Information("MACE Inference Service ready ✓");
        startLog.Information("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    });

    lifetime.ApplicationStopping.Register(() =>
        startLog.Information("MACE Inference Service stopping..."));

    lifetime.ApplicationStopped.Register(() =>
        startLog.Information("MACE Inference Service stopped."));

    app.Run();
    return 0;
}
catch (Exception ex)
{
    startLog.Fatal(ex, "MACE Inference Service terminated unexpectedly.");
    return 1;
}
finally
{
    Log.CloseAndFlush();
}

// ─────────────────────────────────────────────────────────────────────────────
// Strips the namespace prefix from SourceContext so log output shows just
// the class name, e.g. "MaceInferenceGrpcService" instead of
// "MACE.Services.MaceInferenceGrpcService".
// ─────────────────────────────────────────────────────────────────────────────
sealed class ShortClassNameEnricher : ILogEventEnricher
{
    public void Enrich(LogEvent logEvent, ILogEventPropertyFactory factory)
    {
        var full = logEvent.Properties.TryGetValue("SourceContext", out var prop)
            ? prop.ToString().Trim('"')
            : string.Empty;

        var dot       = full.LastIndexOf('.');
        var shortName = dot >= 0 ? full[(dot + 1)..] : full;

        logEvent.AddOrUpdateProperty(factory.CreateProperty("ShortContext", shortName));
    }
}
