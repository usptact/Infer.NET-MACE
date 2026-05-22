# Contributing to Infer.NET MACE

Contributions are welcome. For significant changes — new features, model modifications, breaking API changes — please open an issue first to discuss the approach before writing code.

## Getting Started

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Restore dependencies:
   ```bash
   dotnet restore
   ```
3. Build to confirm a clean baseline:
   ```bash
   dotnet build
   ```
4. Run the sample to confirm the model produces sensible output:
   ```bash
   dotnet run --project MACE -- MACE/sample_data.txt
   ```

## Making Changes

### Code Style

This project follows modern C# conventions:

- Enable nullable reference types and resolve all nullability warnings before submitting
- Prefer `var` for local variables where the type is evident from context
- Use expression-bodied members for simple single-expression properties and methods
- Include XML doc comments (`/// <summary>`) on all `public` and `protected` members
- Follow the existing `PascalCase` / `_camelCase` naming conventions for members and fields

### Probabilistic Model Changes

The core model lives in `MACEBase` and `MACETrain`. If you modify it:

- Verify that inference still converges on `sample_data.txt` and produces qualitatively correct results (Worker 7 identified as spammer, uncertain items flagged)
- Be aware that VMP is sensitive to initialisation — label initialisation randomises the starting point to break symmetry, so run a few times to confirm stability
- If you change the number of inference iterations needed for convergence, update the default in `Program.cs` and the README

### Input Validation

User-facing errors should be thrown as `InvalidOperationException` with a clear, actionable message. They are caught at the top level and printed without a stack trace. Reserve other exception types for genuinely unexpected failures.

### Data Format Compatibility

`CsvReader` enforces:
- One column per worker, one row per item
- Integer labels forming a contiguous range starting from 0
- At least 3 workers per item (warning, not an error)

Changes that alter these constraints must be reflected in the README data format documentation.

## Submitting a Pull Request

1. Ensure `dotnet build` produces **0 errors**
2. Run the sample end-to-end and include the console output in the PR description if the results change
3. Update `README.md` if your change affects usage, CLI flags, data format requirements, or the API example
4. Keep commits focused — one logical change per commit with a descriptive message
5. Open the PR against `master`

## Reporting Issues

Please include:

- .NET SDK version (`dotnet --version`)
- Operating system
- The input file or a minimal reproducing CSV snippet
- The full console output including any error message

## Project Structure

```
MACE/
  MACEBase.cs       # Abstract probabilistic model (priors, latent variables)
  MACETrain.cs      # Observation model and VMP inference
  ModelData.cs      # Container for prior/posterior distributions
  CsvReader.cs      # CSV parsing and input validation
  Program.cs        # CLI entry point and output formatting
  sample_data.txt   # Sample annotation matrix for manual testing
MACE.sln
```

## License

By contributing you agree that your changes will be licensed under the [MIT License](LICENSE) that covers this project.
