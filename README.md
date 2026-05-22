# MACE: Multi-Annotator Competence Estimation

[![.NET](https://img.shields.io/badge/.NET-10.0-blue.svg)](https://dotnet.microsoft.com/download/dotnet/10.0)
[![Infer.NET](https://img.shields.io/badge/Infer.NET-0.4.2504.701-purple.svg)](https://dotnet.github.io/infer/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modern .NET 10.0 implementation of the MACE (Multi-Annotator Competence Estimation) algorithm using Microsoft's Infer.NET probabilistic programming framework. This implementation is based on the research paper "Learning Whom to Trust with MACE" by Dirk Hovy et al., published at NAACL 2013.

## Table of Contents

- [Problem Statement](#problem-statement)
- [How MACE Solves the Problem](#how-mace-solves-the-problem)
- [Model Design and Assumptions](#model-design-and-assumptions)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Understanding the Output](#understanding-the-output)
- [Example](#example)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Problem Statement

Crowdsourcing has become a popular approach for collecting large amounts of labeled data for machine learning tasks. However, crowdsourcing introduces several challenges:

1. **Worker Quality Variation**: Different workers have different levels of expertise and reliability
2. **Spammer Detection**: Some workers may provide random or malicious labels to maximize their earnings
3. **Missing Annotations**: Not all workers annotate all items, leading to incomplete data
4. **Ground Truth Uncertainty**: The true labels for items are often unknown, making it difficult to assess worker quality

Traditional approaches like majority voting fail to account for worker reliability and can be biased by spammers or low-quality workers.

## How MACE Solves the Problem

MACE addresses these challenges through a probabilistic graphical model that simultaneously:

1. **Estimates True Labels**: Infers the most likely true label for each item based on all available annotations
2. **Models Worker Reliability**: Learns a spammer probability for each worker-item pair
3. **Handles Missing Data**: Naturally handles incomplete annotation matrices
4. **Provides Uncertainty Quantification**: Returns probability distributions rather than point estimates

The key insight is that reliable workers should agree with each other on easy items, while spammers will show random or biased behavior patterns.

## Model Design and Assumptions

### Probabilistic Model

MACE uses a hierarchical Bayesian model with the following components:

#### Model Variables

- **T[i]**: True label for item i (discrete distribution over categories)
- **S[i,j]**: Spammer indicator for worker j on item i (Bernoulli)
- **θ[j]**: Worker j's spammer probability (Beta distribution)
- **φ[j]**: Worker j's label preferences when spamming (Dirichlet distribution)

#### Model Structure

```
For each item i:
    T[i] ~ DiscreteUniform(numCategories)
    
    For each worker j:
        S[i,j] ~ Bernoulli(θ[j])
        
        If S[i,j] = 0 (not spamming):
            A[i,j] = T[i]  // Use true label
        Else (spamming):
            A[i,j] ~ Discrete(φ[j])  // Use spammer's preference
```

#### Key Assumptions

1. **Conditional Independence**: Given the true label and spammer indicators, annotations are independent
2. **Worker Consistency**: A worker's spammer behavior is consistent across items (modeled by θ[j])
3. **Spammer Preferences**: When spamming, workers have consistent label preferences (modeled by φ[j])
4. **Uniform Priors**: We use uninformative priors to let the data drive the inference

### Inference

The model uses variational message passing (VMP) for approximate Bayesian inference, which is efficient and scales well to large datasets.

## Features

- ✅ **Modern .NET 10.0**: Built with the latest .NET framework
- ✅ **Latest Infer.NET**: Uses the most recent version (0.4.2504.701)
- ✅ **Robust Error Handling**: Comprehensive input validation and error reporting
- ✅ **XML Documentation**: Fully documented API with IntelliSense support
- ✅ **Cross-Platform**: Runs on Windows, macOS, and Linux
- ✅ **Memory Efficient**: Proper resource management with IDisposable pattern
- ✅ **Type Safe**: Nullable reference types and modern C# features

## Prerequisites

- [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) or later
- Windows, macOS, or Linux

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Infer.NET-MACE.git
cd Infer.NET-MACE
```

2. Restore dependencies:
```bash
dotnet restore
```

3. Build the project:
```bash
dotnet build
```

## Usage

### Basic Usage

Run MACE on a CSV file containing annotation data:

```bash
dotnet run --project MACE -- sample_data.txt
```

### Command Line Options

```bash
MACE.exe <CSV_FILE>
```

**Parameters:**
- `<CSV_FILE>`: Path to the CSV file containing annotation data

**Output Files:**
- `<input_name>_item_labels.csv`: Inferred label probabilities for each item
- `<input_name>_worker_spammer_probs.csv`: Spammer probabilities for each worker-item pair

**Example:**
```bash
dotnet run --project MACE -- MACE/sample_data.txt
```

This will generate:
- `MACE/sample_data_item_labels.csv`
- `MACE/sample_data_worker_spammer_probs.csv`

## Data Format

The input CSV file should follow this format:

### Header Row
The first row contains worker identifiers (column names). Worker names can be arbitrary strings.

### Data Rows
Each subsequent row represents one item to be annotated. Columns correspond to workers, and cells contain:
- **Integer values** (0, 1, 2, ...): The annotation label for that worker-item pair
- **Empty cells**: Missing annotations (worker did not annotate this item)

### Data Quality Requirements

MACE enforces the following data quality requirements:

1. **Minimum Worker Coverage**: Each work item must be seen by at least 3 different workers for reliable inference

### Validation Process

During data loading, MACE automatically:
- **Checks worker coverage**: Reports items with fewer than 3 workers as warnings
- **Provides summary**: Shows detailed validation messages in the console output

### Example

```csv
w1,w2,w3,w4,w5,w6,w7,w8
0,,,1,0,,,0
,2,,,1,,,0
1,1,,,,,0,1
,1,,1,,,,1
1,,,,1,1,0,
,2,1,2,,,,
0,0,2,,,,0,
1,,,,0,0,0,
,1,1,,,0,,
2,0,2,,,2,,
```

**Interpretation:**
- 8 workers (w1 through w8)
- 10 items to be annotated
- 3 label categories (0, 1, 2)
- Missing annotations are represented by empty cells
- Worker 7 appears to be a spammer (always provides label 0)

## Understanding the Output

MACE produces two CSV output files:

### 1. Item Labels CSV (`<input_name>_item_labels.csv`)

Contains the posterior distribution over possible labels for each item:

```csv
Item,Label_0_Probability,Label_1_Probability,Label_2_Probability,Most_Probable_Label,Confidence
Item_1,0.947293,0.043987,0.008720,Label_0,0.947293
Item_2,0.368018,0.324444,0.307538,Label_0,0.368018
Item_3,0.020812,0.968046,0.011142,Label_1,0.968046
```

**Columns:**
- `Item`: Item identifier
- `Label_X_Probability`: Probability that the item belongs to label category X
- `Most_Probable_Label`: The label with highest probability
- `Confidence`: The probability of the most probable label

**Interpretation:**
- Item 1: 94.7% probability of being label 0, 4.4% label 1, 0.9% label 2
- Item 2: Uncertain between all three labels (high entropy)
- Item 3: 96.8% probability of being label 1

### 2. Worker Spammer Probabilities CSV (`<input_name>_worker_spammer_probs.csv`)

Contains the probability that each worker is spamming on each item:

```csv
Item,Worker,Spammer_Probability
Item_1,Worker_1,0.198094
Item_1,Worker_4,0.964733
Item_1,Worker_5,0.288048
Item_1,Worker_8,0.281414
```

Only rows where the worker actually annotated the item are included; missing annotation pairs have no meaningful spammer posterior.

**Columns:**
- `Item`: Item identifier
- `Worker`: Worker identifier
- `Spammer_Probability`: Probability that the worker is spamming on this item

**Interpretation:**
- Values close to 1.0 indicate likely spamming behavior
- Values close to 0.0 indicate reliable annotation
- Worker 7 shows consistently high spammer probabilities (0.69+)

### Interpretation Guidelines

1. **Item Difficulty**: High entropy in label distributions indicates difficult items
2. **Worker Quality**: Consistently high spammer probabilities suggest unreliable workers
3. **Confidence**: Sharp probability distributions indicate high confidence in predictions
4. **Missing Data**: The model naturally handles incomplete annotation matrices
5. **Data Quality**: Pay attention to validation warnings - items with insufficient workers may have unreliable predictions

## Data Quality Best Practices

### Recommended Data Collection

1. **Worker Coverage**: Aim for at least 3-5 workers per item for reliable inference
2. **Balanced Design**: Try to have roughly equal numbers of annotations per worker
3. **Quality Control**: Include some gold standard items to validate worker quality

### Common Issues and Solutions

1. **Insufficient Workers**: 
   - **Problem**: Items with <3 workers have unreliable predictions
   - **Solution**: Collect more annotations for these items or exclude them

2. **Imbalanced Coverage**:
   - **Problem**: Some workers annotate many items, others few
   - **Solution**: This is acceptable, but consider worker reliability scores

### Validation Messages

MACE provides detailed validation messages to help you understand data quality:

- **WARNING**: Items with insufficient worker coverage
- **Statistics**: Summary of data dimensions and coverage

## Example

Let's walk through a complete example using the provided sample data:

```bash
# Run MACE on the sample data
dotnet run --project MACE -- MACE/sample_data.txt
```

**Sample Console Output:**
```
=== MACE: Multi-Annotator Competence Estimation ===

Input file: MACE/sample_data.txt
Output files:
  Item labels: MACE/sample_data_item_labels.csv
  Spammer probabilities: MACE/sample_data_worker_spammer_probs.csv

Reading input data...
*** DATA STATISTICS ***
Number of items: 10
Number of workers: 8
Number of categories: 3

Initializing MACE model priors...
Creating probabilistic model...
Running probabilistic inference...
Compiling model...done.
Iterating:
.........|.........|.........|.........|.........| 50
Writing results to CSV files...

*** INFERENCE COMPLETED SUCCESSFULLY ***
Results written to:
  - MACE/sample_data_item_labels.csv
  - MACE/sample_data_worker_spammer_probs.csv
```

**Analysis:**
- Items 1, 3, 4, 5, 6, 7, 10 have confident predictions (high confidence values)
- Item 2 is genuinely difficult (low confidence across all three labels)
- Worker 7 is correctly identified as a spammer (spammer probabilities of 0.69–0.99)
- Worker 8 has mixed reliability — low spammer probability on most items but elevated on Item 2, reflecting genuine uncertainty in that item's label

## API Documentation

### Core Classes

#### `MACEBase`
Abstract base class defining the probabilistic model structure.

```csharp
public abstract class MACEBase
{
    public InferenceEngine InferenceEngine { get; protected set; }
    public virtual void CreateModel();
    public virtual void SetModelData(ModelData modelData);
    public void InitializeLabels(int numItems, int numCategories);
}
```

#### `MACETrain`
Implements the complete MACE model including the observation model.

```csharp
public class MACETrain : MACEBase
{
    public MACETrain(int numWorkers, int numItems, int numCategories);
    public override void CreateModel();
    public ModelData InferModelData(int[][] data);
}
```

#### `ModelData`
Container for model parameters and posterior distributions.

```csharp
public class ModelData
{
    public Beta[] ThetaDist { get; set; }        // Worker spammer probabilities
    public Dirichlet[] PhiDist { get; set; }     // Worker label preferences
    public Discrete[] TDist { get; set; }        // True label distributions
    public Bernoulli[][] SDist { get; set; }     // Spammer indicators
}
```

#### `CsvReader`
Handles reading and parsing CSV annotation files.

```csharp
public class CsvReader : IDisposable
{
    public CsvReader(string fileName);
    public void Read();
    public int[][] GetData();
    public int GetNumWorkers();
    public int GetNumItems();
    public int GetNumCategories();
}
```

### Usage Example

```csharp
using var reader = new CsvReader("annotations.csv");
reader.Read();
var data = reader.GetData();

var trainer = new MACETrain(
    reader.GetNumWorkers(),
    reader.GetNumItems(),
    reader.GetNumCategories() + 1
);

trainer.CreateModel();
trainer.InitializeLabels(reader.GetNumItems(), reader.GetNumCategories() + 1);

var initPriors = new ModelData
{
    ThetaDist = Enumerable.Range(0, reader.GetNumWorkers())
        .Select(_ => new Beta(1, 1)).ToArray(),
    PhiDist = Enumerable.Range(0, reader.GetNumWorkers())
        .Select(_ => new Dirichlet(Enumerable.Repeat(1.0, reader.GetNumCategories() + 1).ToArray()))
        .ToArray()
};

trainer.SetModelData(initPriors);
var posterior = trainer.InferModelData(data);

// Access results
var itemLabels = posterior.TDist;
var spammerProbs = posterior.SDist;
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

This project follows modern C# conventions:
- Use nullable reference types
- Prefer `var` for local variables
- Use expression-bodied members where appropriate
- Include XML documentation for public APIs
- Follow the existing naming conventions

## References

1. **Original MACE Paper**: Dirk Hovy, Taylor Berg-Kirkpatrick, Ashish Vaswani, Eduard Hovy. "Learning Whom to Trust with MACE". Proceedings of NAACL 2013. [PDF](http://www.aclweb.org/anthology/N13-1132)

2. **Infer.NET**: Microsoft Research's probabilistic programming framework. [Website](https://dotnet.github.io/infer/)

3. **Crowdsourcing Quality Control**: For broader context on crowdsourcing quality assessment methods.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a modernized implementation of the original MACE algorithm. While the core probabilistic model remains the same, the codebase has been updated to use .NET 10.0, the latest Infer.NET framework, and modern C# best practices for improved maintainability, performance, and developer experience.