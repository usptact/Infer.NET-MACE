using MACE.Core;
using MACE.Services;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Prometheus;

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddEnvironmentVariables();

builder.Services.Configure<InferenceOptions>(
    builder.Configuration.GetSection("Inference"));

// Pool is a singleton — CreateModel() runs once per slot during construction.
builder.Services.AddSingleton<InferencePool>();
builder.Services.AddSingleton<PriorUpdateService>();

builder.Services.AddGrpc();

// gRPC requires HTTP/2. In a containerised environment TLS is terminated at the
// load-balancer level, so we listen on cleartext HTTP/2 (H2C).
builder.WebHost.ConfigureKestrel(o =>
{
    o.ListenAnyIP(8080, lo => lo.Protocols = HttpProtocols.Http2);
});

var app = builder.Build();

// Force the singleton to construct (and pre-warm the pool) before the first request.
_ = app.Services.GetRequiredService<InferencePool>();

app.MapGrpcService<MaceInferenceGrpcService>();

// Prometheus metrics endpoint (HTTP/1.1 compatible via the gRPC server's HTTP/2 port
// when accessed with a standard client — or move to a sidecar if needed).
app.UseMetricServer(port: 9090);
app.UseHttpMetrics();

app.MapGet("/", () => "MACE Inference Service. Use a gRPC client.");

app.Run();
