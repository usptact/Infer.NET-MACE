# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Restore dependencies
dotnet restore

# Build
dotnet build

# Run on a CSV annotation file
dotnet run --project MACE -- MACE/sample_data.txt
```

There are no automated tests in this project.

## Architecture

This is a single-project .NET 10.0 console application (`MACE/MACE.csproj`) implementing the MACE algorithm ("Learning Whom to Trust with MACE", Hovy et al., NAACL 2013) for crowdsourcing annotation quality estimation. It uses [Microsoft Infer.NET](https://dotnet.github.io/infer/) (`Microsoft.ML.Probabilistic`) for variational message passing (VMP) inference.

### Data flow

1. `CsvReader` parses the annotation matrix (rows = items, columns = workers, cells = integer labels or empty for missing). It validates that each item has ≥3 workers and removes duplicate annotations.
2. `Program.Main` initializes uniform priors (`Beta(1,1)` for spammer rates θ, `Dirichlet(1,...,1)` for spammer label preferences φ), constructs the model, and runs inference.
3. `MACETrain.InferModelData` sets the observed annotation data and calls `InferenceEngine.Infer` for each latent variable.
4. Results are written to two CSV files: item label posteriors and per-(item, worker) spammer probabilities.

### Probabilistic model

- **`MACEBase`** — Abstract base. Declares all Infer.NET `Variable` objects: `_theta` (worker spammer probability, Beta prior), `_phi` (spammer label preference, Dirichlet prior), `_trueLabels` (true label per item), `_spammerIndicators` (Bernoulli per item×worker). `CreateModel()` wires the priors; `SetModelData()` sets observed prior values; `InitializeLabels()` randomizes initial label assignments to break symmetry.
- **`MACETrain : MACEBase`** — Adds the observation model. `CreateModel()` loops over items×workers: each annotation is either the true label (non-spammer path) or drawn from the worker's φ (spammer path), conditioned on the spammer indicator S[i,j]. Missing values (-1) are excluded via `Variable.If`. `InferModelData(int[][] data)` runs inference and returns posteriors.
- **`ModelData`** — Plain container for `Beta[] ThetaDist`, `Dirichlet[] PhiDist`, `Discrete[] TDist`, `Bernoulli[][] SDist`. Used both for priors (input) and posteriors (output).
- **`CsvReader : IDisposable`** — Reads CSV, stores data as `List<int[]>` (missing = -1). `GetNumCategories()` returns `max_label_value`; callers add 1 when passing to the model.

### Important nuance

`GetNumCategories()` returns the highest observed label value (0-based max), not the count. `Program.cs` always passes `reader.GetNumCategories() + 1` to model constructors. The duplicate-detection logic in `HandleDuplicateAnnotations` considers annotations with the same *value* (not same worker column) as duplicates — this is a content-based heuristic, not identity-based.
