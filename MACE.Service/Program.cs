using MACE;
using MACE.Online;
using MACE.Service;
using MACE.Service.Services;
using Microsoft.Extensions.Options;
using Microsoft.ML.Probabilistic.Distributions;
using Prometheus;
using Serilog;

var builder = WebApplication.CreateBuilder(args);

builder.Host.UseSerilog((context, configuration) => configuration
    .ReadFrom.Configuration(context.Configuration)
    .Enrich.FromLogContext()
    .WriteTo.Console());

builder.Services.Configure<ServiceOptions>(builder.Configuration.GetSection(ServiceOptions.SectionName));
builder.Services.AddSingleton<IValidateOptions<ServiceOptions>, ValidateServiceOptions>();

builder.Services.AddSingleton<PriorUpdateService>();

// The pool warms its models here, at startup, so the first request does not pay for compiling the
// inference algorithm.
builder.Services.AddSingleton<IInferencePool>(provider =>
{
    var options = provider.GetRequiredService<IOptions<ServiceOptions>>().Value;
    var logger = provider.GetRequiredService<ILogger<InferencePool>>();

    logger.LogInformation(
        "Warming {PoolSize} model(s) for {Workers} workers and {Categories} categories...",
        options.PoolSize, options.NumWorkers, options.NumCategories);

    return new InferencePool(
        new InferencePoolOptions(
            options.NumWorkers, options.NumCategories, options.PoolSize, options.Iterations, options.Seed),
        message => logger.LogInformation("{Message}", message));
});

builder.Services.AddSingleton(provider =>
{
    var options = provider.GetRequiredService<IOptions<ServiceOptions>>().Value;
    var logger = provider.GetRequiredService<ILogger<BeliefStore>>();
    var updates = provider.GetRequiredService<PriorUpdateService>();

    var priors = LoadPriors(options, logger);
    return new BeliefStore(priors, updates);
});

builder.Services.AddGrpc();

var app = builder.Build();

// Resolve these now. DI builds singletons on first use, so leaving it to the first request would
// hand that caller the model compilation the pool exists to keep off the request path, and would
// defer a bad BeliefStorePath until then too.
_ = app.Services.GetRequiredService<BeliefStore>();
_ = app.Services.GetRequiredService<IInferencePool>();

app.UseRouting();
app.UseHttpMetrics();
app.MapGrpcService<MaceInferenceService>();
app.MapMetrics();

app.MapGet("/", () => Results.Text(
    "MACE inference service. gRPC on this port; Prometheus metrics at /metrics.",
    "text/plain"));

// Persist what the service learned, so a restart does not discard every worker's history.
app.Lifetime.ApplicationStopping.Register(() =>
{
    var options = app.Services.GetRequiredService<IOptions<ServiceOptions>>().Value;
    if (options.BeliefStorePath == null)
    {
        return;
    }

    var logger = app.Services.GetRequiredService<ILogger<BeliefStore>>();
    try
    {
        var store = app.Services.GetRequiredService<BeliefStore>();
        ModelPriorsIo.Save(store.Snapshot(), options.BeliefStorePath);
        logger.LogInformation("Saved worker parameters to {Path}", options.BeliefStorePath);
    }
    catch (Exception ex)
    {
        // Shutdown must not fail because persistence did; the operator needs to see why, though.
        logger.LogError(ex, "Could not save worker parameters to {Path}", options.BeliefStorePath);
    }
});

app.Run();

static ModelPriors LoadPriors(ServiceOptions options, Microsoft.Extensions.Logging.ILogger logger)
{
    if (options.BeliefStorePath != null && File.Exists(options.BeliefStorePath))
    {
        try
        {
            var loaded = ModelPriorsIo.Load(
                options.BeliefStorePath, options.NumWorkers, options.NumCategories);
            logger.LogInformation(
                "Loaded worker parameters from {Path}", options.BeliefStorePath);
            return loaded;
        }
        catch (InvalidOperationException ex)
        {
            // Starting from uniform priors silently would throw away every worker's history and
            // look like nothing was wrong, so refuse instead and let the operator decide.
            throw new InvalidOperationException(
                $"Could not load worker parameters from '{options.BeliefStorePath}'. Fix the file or "
                + "clear Mace:BeliefStorePath to start from uniform priors.", ex);
        }
    }

    if (options.BeliefStorePath != null)
    {
        logger.LogInformation(
            "No parameters at {Path} yet; starting from uniform priors.", options.BeliefStorePath);
    }
    else
    {
        logger.LogWarning(
            "Mace:BeliefStorePath is not set. Worker reliability will be lost when this process exits.");
    }

    var concentration = Enumerable.Repeat(1.0, options.NumCategories).ToArray();
    return new ModelPriors(
        ThetaDist: Enumerable.Range(0, options.NumWorkers).Select(_ => new Beta(1, 1)).ToArray(),
        PhiDist: Enumerable.Range(0, options.NumWorkers).Select(_ => new Dirichlet(concentration)).ToArray());
}

/// <summary>Fails startup on unusable configuration rather than at the first request.</summary>
internal sealed class ValidateServiceOptions : IValidateOptions<ServiceOptions>
{
    public ValidateOptionsResult Validate(string? name, ServiceOptions options)
    {
        try
        {
            options.Validate();
            return ValidateOptionsResult.Success;
        }
        catch (InvalidOperationException ex)
        {
            return ValidateOptionsResult.Fail(ex.Message);
        }
    }
}

/// <summary>Entry point marker so integration tests can reference this host.</summary>
public partial class Program;
