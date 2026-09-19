# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Restore dependencies
dotnet restore

# Build
dotnet build

# Run the test suite
dotnet test

# Run on a CSV annotation file
dotnet run --project MACE -- MACE/sample_data.txt

# Optional flags: iteration count and a seed for reproducible runs
dotnet run --project MACE -- MACE/sample_data.txt --iterations 100 --seed 42
```

Tests live in `MACE.Tests` (xUnit). `MACE/sample_data.txt` and `MACE/true_labels.txt` are linked into the test output and serve as the accuracy regression corpus.

## Architecture

This is a single-project .NET 10.0 console application (`MACE/MACE.csproj`) implementing the MACE algorithm ("Learning Whom to Trust with MACE", Hovy et al., NAACL 2013) for crowdsourcing annotation quality estimation. It uses [Microsoft Infer.NET](https://dotnet.github.io/infer/) (`Microsoft.ML.Probabilistic`) for expectation propagation (EP) inference.

### Data flow

1. `CsvReader` parses the annotation matrix (rows = items, columns = workers, cells = integer labels or empty for missing) and validates it.
2. `CsvReader.GetSparseData()` converts the matrix into `SparseAnnotations` — only the observed (item, worker, label) triples.
3. `Program.Main` builds uniform priors (`Beta(1,1)` for spammer rates θ, `Dirichlet(1,...,1)` for spammer label preferences φ) and constructs `MACETrain`, which builds the model in its constructor.
4. `MACETrain.InferModelData(annotations, priors)` binds the observed data and calls `InferenceEngine.Infer` for each latent variable, returning a `ModelPosterior`.
5. Results are written to three CSV files: item label posteriors, per-annotation spammer probabilities, and per-worker competence.

### Probabilistic model

- **`MACEBase`** — Abstract base. Declares the shared Infer.NET `Variable` objects: `_theta` (per-worker spammer probability, Beta prior), `_phi` (per-worker label preference when spamming, Dirichlet prior), `_trueLabels` (true label per item), and the item/worker `Range`s. `CreateModel()` wires the priors and creates the engine; `SetModelData(ModelPriors)` binds observed prior values; `InitializeLabels()` assigns random point-mass labels to break symmetry on multimodal data.
- **`MACETrain : MACEBase`** — Adds the observation model over a **sparse jagged representation**. `_obsRange` is nested inside `_itemRange` and sized by `_numObsPerItem[item]`, so the inner loop runs only over workers who actually annotated that item. Per observation: `S ~ Bernoulli(theta[workerIdx])`; under `Variable.If(S == false)` the annotation *is* the true label, under `Variable.If(S == true)` it is drawn from `Discrete(phi[workerIdx])`. There are no sentinel values and no missingness guards — absent annotations simply are not in the arrays. Observed arrays carry `DoNotInfer`.
  - Call order is enforced structurally: the constructor validates dimensions, builds the arrays, calls `CreateModel()`, and sets the iteration count, so there is no half-built state. `InferModelData` is the only other entry point.
- **`ModelPriors` / `ModelPosterior`** (both in `ModelData.cs`) — `ModelPosterior` *inherits* from `ModelPriors`, so a completed run's output can be passed straight back in as priors for incremental/online learning. `ModelPosterior` adds `Discrete[] TDist` (item label posteriors) and `Bernoulli[][] SDist` (spammer indicators).
- **`SparseAnnotations`** (in `CsvReader.cs`) — `int[][] WorkerIndices` and `int[][] Labels`, parallel per item.
- **`CsvReader : IDisposable`** — Reads CSV into a dense `List<int[]>` (missing = -1) and exposes it as `SparseAnnotations`. Coverage problems are advisory: items with no annotations and items with fewer than 3 are reported separately through `GetValidationMessages()` and still take part in inference.
- **`ModelPriorsIo`** — Persists `ModelPriors` as a small text file, which is what makes the incremental path (`--save-priors` / `--load-priors`) reachable across processes. A worker or category count that disagrees with the current dataset is rejected rather than reinterpreted.

### Important nuances

- **Items and workers with no annotations are still inferred.** Their posteriors are the priors, so the label reported for such an item is an argmax over a uniform distribution and the competence reported for such a worker is whatever the prior said — neither is a measurement. `CheckWorkerCoverage` warns about both, separately from thin item coverage. The same caveat applies to any exactly tied posterior.
- **`CsvReader.Read()` may be called once per instance.** The stream is consumed, so a second call throws rather than reporting an empty file and doubling the item count.
- **Carrying priors forward is only valid onto new annotations.** Re-running the same data against its own posterior counts that evidence twice and reports false confidence. `ModelPriorsIo` cannot detect this; the docs warn instead.
- **`SDist` is indexed by annotation slot, not worker.** `SDist[item][k]` is parallel to `annotations.WorkerIndices[item][k]`; the actual worker index is `WorkerIndices[item][k]`. `WriteSpammerProbabilitiesToCsv` needs the annotations alongside the posterior for exactly this reason.
- **`GetNumCategories()` returns the category *count*** (`max(label) + 1`), not the max label value. `Program.cs` passes it to the model constructor unchanged — do not add 1.
- **Two validation passes with different severity.** `CheckWorkerCoverage()` only *warns* (messages surface via `GetValidationMessages()`) when an item has fewer than 3 annotators; inference still runs. `ValidateLabelRange()` *throws* when the observed labels have a gap (e.g. `{0, 2}`), because a phantom category would silently skew inference.
- **Reproducibility.** Runs are already deterministic across fresh processes, because Infer.NET's `Rand` starts from a fixed default seed. `--seed` perturbs the symmetry-breaking initialisation, and it only has a visible effect where the data does not determine the answer on its own: `MACE/sample_data.txt` converges to a unique fixed point, so every seed gives identical output there and that file cannot be used to test the flag. `TestSupport.PerfectlyTiedCorpus` can.
- **Initialisation must be an observed value, not a literal.** `MACEBase` passes it through the `TInit` variable. Handing a literal distribution to `InitialiseTo` compiles the values into the generated algorithm as constants, and the cached algorithm then keeps the *first* run's initialisation — so the seed works from the CLI (one run per process) and silently stops working when inference is called repeatedly in one process, which is exactly the incremental-learning path. `Seed_AffectsPosterior_WithinASingleProcess` guards this.
- **The engine is EP, and that is load-bearing.** `MACEBase.CreateModel()` constructs `new InferenceEngine(new ExpectationPropagation())` explicitly. Switching to VMP throws "The model has zero probability": the non-spammer branch assigns `_observedLabels = _trueLabels` deterministically, so under VMP each annotator contributes a point mass at its own label and any disagreement multiplies to zero. Changing engines means reformulating the observation model as a soft confusion matrix.
