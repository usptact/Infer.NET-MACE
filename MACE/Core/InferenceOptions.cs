namespace MACE.Core;

public class InferenceOptions
{
    public int NumSensorTypes          { get; set; } = 7;
    public int NumCategories           { get; set; } = 5;
    public int PoolSize                { get; set; } = 4;
    public int MinSensorsForInference  { get; set; } = 2;
    public int PoolAcquireTimeoutMs    { get; set; } = 5000;
}
