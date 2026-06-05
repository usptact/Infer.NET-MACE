using FluentAssertions;
using Serilog.Core;
using Xunit;
using Serilog.Events;
using Serilog.Parsing;

namespace MACE.Tests.Logging;

/// <summary>
/// Tests for the internal ShortClassNameEnricher defined in MACE/Program.cs.
/// Accessible via [assembly: InternalsVisibleTo("MACE.Tests")] in AssemblyInfo.cs.
/// </summary>
public class ShortClassNameEnricherTests
{
    // ── Helpers ───────────────────────────────────────────────────────────────

    // Minimal ILogEventPropertyFactory that creates real LogEventProperty objects
    // so AddOrUpdateProperty() on the LogEvent succeeds.
    private sealed class StubFactory : ILogEventPropertyFactory
    {
        public LogEventProperty CreateProperty(string name, object? value, bool destructureObjects = false)
            => new(name, new ScalarValue(value));
    }

    private static readonly ILogEventPropertyFactory Factory = new StubFactory();
    private static readonly MessageTemplate EmptyTemplate = new MessageTemplateParser().Parse(string.Empty);

    private static LogEvent CreateLogEvent(string? sourceContext = null)
    {
        var properties = new List<LogEventProperty>();
        if (sourceContext is not null)
            properties.Add(new LogEventProperty("SourceContext", new ScalarValue(sourceContext)));

        return new LogEvent(
            DateTimeOffset.UtcNow,
            LogEventLevel.Information,
            exception: null,
            messageTemplate: EmptyTemplate,
            properties: properties);
    }

    private static string? ShortContext(LogEvent logEvent)
    {
        if (!logEvent.Properties.TryGetValue("ShortContext", out var val))
            return null;
        return ((ScalarValue)val).Value as string;
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Enrich_SingleNamespace_ExtractsClassName()
    {
        var enricher = new ShortClassNameEnricher();
        var logEvent = CreateLogEvent("MACE.Program");

        enricher.Enrich(logEvent, Factory);

        ShortContext(logEvent).Should().Be("Program");
    }

    [Fact]
    public void Enrich_DeepNamespace_ExtractsLastSegmentOnly()
    {
        var enricher = new ShortClassNameEnricher();
        var logEvent = CreateLogEvent("MACE.Services.MaceInferenceGrpcService");

        enricher.Enrich(logEvent, Factory);

        ShortContext(logEvent).Should().Be("MaceInferenceGrpcService");
    }

    [Fact]
    public void Enrich_NoNamespace_ReturnsFullString()
    {
        var enricher = new ShortClassNameEnricher();
        var logEvent = CreateLogEvent("StandaloneClass");

        enricher.Enrich(logEvent, Factory);

        ShortContext(logEvent).Should().Be("StandaloneClass");
    }

    [Fact]
    public void Enrich_MissingSourceContext_SetsEmptyShortContext()
    {
        var enricher = new ShortClassNameEnricher();
        var logEvent = CreateLogEvent(sourceContext: null); // no SourceContext property

        enricher.Enrich(logEvent, Factory);

        // Property must still be added (with empty string)
        logEvent.Properties.Should().ContainKey("ShortContext");
        ShortContext(logEvent).Should().Be(string.Empty);
    }

    [Fact]
    public void Enrich_AlwaysAddsPropertyNamed_ShortContext()
    {
        var enricher = new ShortClassNameEnricher();
        var logEvent = CreateLogEvent("A.B.C.D");

        enricher.Enrich(logEvent, Factory);

        logEvent.Properties.Should().ContainKey("ShortContext");
    }
}
