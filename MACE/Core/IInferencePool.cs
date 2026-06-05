namespace MACE.Core;

/// <summary>
/// Abstraction over the pre-warmed MACETrain instance pool.
/// Extracted so the gRPC service can be unit-tested with a mock pool.
/// </summary>
public interface IInferencePool
{
    int Available { get; }
    int Total     { get; }

    /// <summary>
    /// Acquires exclusive ownership of a pooled <see cref="MACETrain"/> instance.
    /// Dispose the returned <see cref="PooledInference"/> to release the slot.
    /// </summary>
    Task<PooledInference> AcquireAsync(CancellationToken ct = default);
}
