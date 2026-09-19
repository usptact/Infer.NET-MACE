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

# Optional flags: iteration count and a seed for reproducible runs
dotnet run --project MACE -- MACE/sample_data.txt --iterations 100 --seed 42
```

There are no automated tests in this project.

## Architecture

This is a single-project .NET 10.0 console application (`MACE/MACE.csproj`) implementing the MACE algorithm ("Learning Whom to Trust with MACE", Hovy et al., NAACL 2013) for crowdsourcing annotation quality estimation. It uses [Microsoft Infer.NET](https://dotnet.github.io/infer/) (`Microsoft.ML.Probabilistic`) for variational message passing (VMP) inference.

### Data flow

1. `CsvReader` parses the annotation matrix (rows = items, columns = workers, cells = integer labels or empty for missing) and validates it.
2. `CsvReader.GetSparseData()` converts the matrix into `SparseAnnotations` — only the observed (item, worker, label) triples.
3. `Program.Main` builds uniform priors (`Beta(1,1)` for spammer rates θ, `Dirichlet(1,...,1)` for spammer label preferences φ) and constructs `MACETrain`, which builds the model in its constructor.
4. `MACETrain.InferModelData(annotations, priors)` binds the observed data and calls `InferenceEngine.Infer` for each latent variable, returning a `ModelPosterior`.
5. Results are written to two CSV files: item label posteriors and per-annotation spammer probabilities.

### Probabilistic model

- **`MACEBase`** — Abstract base. Declares the shared Infer.NET `Variable` objects: `_theta` (per-worker spammer probability, Beta prior), `_phi` (per-worker label preference when spamming, Dirichlet prior), `_trueLabels` (true label per item), and the item/worker `Range`s. `CreateModel()` wires the priors and creates the engine; `SetModelData(ModelPriors)` binds observed prior values; `InitializeLabels()` assigns random point-mass labels to break the symmetry VMP would otherwise get stuck in.
- **`MACETrain : MACEBase`** — Adds the observation model over a **sparse jagged representation**. `_obsRange` is nested inside `_itemRange` and sized by `_numObsPerItem[item]`, so the inner loop runs only over workers who actually annotated that item. Per observation: `S ~ Bernoulli(theta[workerIdx])`; under `Variable.If(S == false)` the annotation *is* the true label, under `Variable.If(S == true)` it is drawn from `Discrete(phi[workerIdx])`. There are no sentinel values and no missingness guards — absent annotations simply are not in the arrays. Observed arrays carry `DoNotInfer`.
  - Call order is enforced structurally: the constructor validates dimensions, builds the arrays, calls `CreateModel()`, and sets the iteration count, so there is no half-built state. `InferModelData` is the only other entry point.
- **`ModelPriors` / `ModelPosterior`** (both in `ModelData.cs`) — `ModelPosterior` *inherits* from `ModelPriors`, so a completed run's output can be passed straight back in as priors for incremental/online learning. `ModelPosterior` adds `Discrete[] TDist` (item label posteriors) and `Bernoulli[][] SDist` (spammer indicators).
- **`SparseAnnotations`** (in `CsvReader.cs`) — `int[][] WorkerIndices` and `int[][] Labels`, parallel per item.
- **`CsvReader : IDisposable`** — Reads CSV into a dense `List<int[]>` (missing = -1) and exposes it as `SparseAnnotations`.

### Important nuances

- **`SDist` is indexed by annotation slot, not worker.** `SDist[item][k]` is parallel to `annotations.WorkerIndices[item][k]`; the actual worker index is `WorkerIndices[item][k]`. `WriteSpammerProbabilitiesToCsv` needs the annotations alongside the posterior for exactly this reason.
- **`GetNumCategories()` returns the category *count*** (`max(label) + 1`), not the max label value. `Program.cs` passes it to the model constructor unchanged — do not add 1.
- **Two validation passes with different severity.** `CheckWorkerCoverage()` only *warns* (messages surface via `GetValidationMessages()`) when an item has fewer than 3 annotators; inference still runs. `ValidateLabelRange()` *throws* when the observed labels have a gap (e.g. `{0, 2}`), because a phantom category would silently skew inference.
- **Reproducibility.** Inference is only deterministic when `--seed` is passed; it flows to `MACETrain`, which calls `Rand.Restart` before randomizing the initial label assignment.
