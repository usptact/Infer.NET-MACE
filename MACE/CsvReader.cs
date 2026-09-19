namespace MACE
{
    /// <summary>
    /// Sparse representation of crowdsourcing annotations.
    /// For each item, only the workers who actually annotated it are stored.
    /// </summary>
    /// <param name="WorkerIndices">
    /// <c>WorkerIndices[item][k]</c> — index of the worker who gave the k-th annotation for this item.
    /// </param>
    /// <param name="Labels">
    /// <c>Labels[item][k]</c> — label given by <c>WorkerIndices[item][k]</c>. Parallel to
    /// <paramref name="WorkerIndices"/>.
    /// </param>
    public record SparseAnnotations(
        int[][] WorkerIndices,
        int[][] Labels
    );

    /// <summary>
    /// Reads and parses CSV files containing crowdsourcing annotation data.
    /// The CSV format expects a header row with worker names, followed by rows of annotation data.
    /// Missing annotations are represented by empty cells.
    ///
    /// Coverage expectations are advisory, not enforced: items annotated by fewer than 3 workers,
    /// and items with no annotations at all, are reported through <see cref="GetValidationMessages"/>
    /// and still take part in inference.
    /// </summary>
    public class CsvReader : IDisposable
    {
        private readonly StreamReader _streamReader;
        private List<int[]>? _dataList;
        private int _numWorkers;
        private int _numItems;
        private int _numCategories;
        private bool _disposed = false;
        private bool _hasRead = false;
        private readonly List<string> _validationMessages = new();

        /// <summary>
        /// Initializes a new instance of the CsvReader class.
        /// </summary>
        /// <param name="fileName">Path to the CSV file to read.</param>
        /// <exception cref="FileNotFoundException">Thrown when the specified file does not exist.</exception>
        /// <exception cref="ArgumentException">Thrown when fileName is null or empty.</exception>
        public CsvReader(string fileName)
        {
            if (string.IsNullOrWhiteSpace(fileName))
            {
                throw new ArgumentException("File name cannot be null or empty.", nameof(fileName));
            }

            if (!File.Exists(fileName))
            {
                throw new FileNotFoundException($"The file '{fileName}' does not exist.");
            }

            try
            {
                _streamReader = new StreamReader(fileName);
                _dataList = null;
                _numWorkers = 0;
                _numItems = 0;
                _numCategories = 0;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to open file '{fileName}': {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Reads and parses the CSV file, populating internal data structures.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the file format is invalid, when reading fails, or when <see cref="Read"/> has
        /// already been called on this instance.
        /// </exception>
        public void Read()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(CsvReader));
            }

            // The underlying stream is consumed by the first pass, so a second call would otherwise
            // fail as though the file were empty and leave the item count doubled.
            if (_hasRead)
            {
                throw new InvalidOperationException(
                    "Read() has already been called on this CsvReader. Create a new instance to read the file again.");
            }

            _hasRead = true;

            try
            {
                // Read header to determine number of workers
                string? headerLine = _streamReader.ReadLine();
                if (headerLine == null)
                {
                    throw new InvalidOperationException("CSV file is empty or has no header.");
                }

                string[] header = headerLine.Split(',');
                _numWorkers = header.Length;

                if (_numWorkers <= 0)
                {
                    throw new InvalidOperationException("CSV file has no workers (columns).");
                }

                _dataList = new List<int[]>();

                // Read data rows
                while (!_streamReader.EndOfStream)
                {
                    string? line = _streamReader.ReadLine();
                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue; // Skip empty lines
                    }

                    string[] fields = line.Split(',');
                    
                    // Ensure all rows have the same number of columns
                    if (fields.Length != _numWorkers)
                    {
                        throw new InvalidOperationException(
                            $"Row {_numItems + 1} has {fields.Length} columns, but expected {_numWorkers}.");
                    }

                    int[] row = new int[_numWorkers];
                    for (int i = 0; i < fields.Length; i++)
                    {
                        if (string.IsNullOrWhiteSpace(fields[i]))
                        {
                            // Missing data is marked with -1
                            row[i] = -1;
                        }
                        else
                        {
                            if (!int.TryParse(fields[i].Trim(), out int value))
                            {
                                throw new InvalidOperationException(
                                    $"Invalid value '{fields[i]}' in row {_numItems + 1}, column {i + 1}. Expected integer or empty value.");
                            }

                            if (value < 0)
                            {
                                throw new InvalidOperationException(
                                    $"Invalid label value {value} in row {_numItems + 1}, column {i + 1}. Label values must be non-negative integers.");
                            }

                            row[i] = value;
                            _numCategories = Math.Max(_numCategories, value + 1);
                        }
                    }

                    _dataList.Add(row);
                    _numItems++;
                }

                if (_numItems == 0)
                {
                    throw new InvalidOperationException("CSV file contains no data rows.");
                }

                if (_numCategories == 0)
                {
                    throw new InvalidOperationException("No valid label categories found in the data.");
                }

                CheckWorkerCoverage();
                ValidateLabelRange();
            }
            catch (Exception ex) when (!(ex is InvalidOperationException))
            {
                throw new InvalidOperationException($"Error reading CSV file: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Returns the parsed data as a two-dimensional array.
        /// </summary>
        /// <returns>A <see cref="SparseAnnotations"/> containing only the observed (item, worker, label) triples.</returns>
        /// <exception cref="InvalidOperationException">Thrown when Read() has not been called.</exception>
        public SparseAnnotations GetSparseData()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(CsvReader));
            if (_dataList == null)
                throw new InvalidOperationException("Read() must be called before GetSparseData().");

            var workerIndices = new int[_numItems][];
            var labels = new int[_numItems][];

            for (int item = 0; item < _numItems; item++)
            {
                var itemWorkers = new List<int>();
                var itemLabels = new List<int>();

                for (int worker = 0; worker < _numWorkers; worker++)
                {
                    if (_dataList[item][worker] != -1)
                    {
                        itemWorkers.Add(worker);
                        itemLabels.Add(_dataList[item][worker]);
                    }
                }

                workerIndices[item] = itemWorkers.ToArray();
                labels[item] = itemLabels.ToArray();
            }

            return new SparseAnnotations(workerIndices, labels);
        }

        /// <summary>
        /// Gets the number of workers (columns) in the dataset.
        /// </summary>
        public int GetNumWorkers() => _numWorkers;

        /// <summary>
        /// Gets the number of items (rows) in the dataset.
        /// </summary>
        public int GetNumItems() => _numItems;

        /// <summary>
        /// Gets the number of distinct label categories in the dataset.
        /// Labels are expected to be contiguous integers starting from 0, so this equals max(label) + 1.
        /// </summary>
        public int GetNumCategories() => _numCategories;

        /// <summary>
        /// Gets the validation messages from data quality checks.
        /// </summary>
        public IReadOnlyList<string> GetValidationMessages() => _validationMessages;

        /// <summary>
        /// Checks annotation coverage per item and emits warnings for thin or absent coverage.
        /// </summary>
        /// <remarks>
        /// Items with no annotations at all are reported separately from merely thin ones. They are
        /// still inferred, but nothing in the data constrains them: their posterior is the prior, and
        /// the label the output CSV reports for them is an artefact of taking an argmax over a
        /// uniform distribution rather than a conclusion drawn from any annotation.
        ///
        /// Workers who annotated nothing are reported for the same reason: their competence is
        /// whatever the prior said, which must not be read as a measurement.
        /// </remarks>
        private void CheckWorkerCoverage()
        {
            if (_dataList == null)
            {
                return;
            }

            _validationMessages.Clear();

            var itemsWithNoWorkers = new List<int>();
            var itemsWithInsufficientWorkers = new List<int>();

            for (int item = 0; item < _numItems; item++)
            {
                int workerCount = 0;
                for (int worker = 0; worker < _numWorkers; worker++)
                {
                    if (_dataList[item][worker] != -1)
                    {
                        workerCount++;
                    }
                }

                if (workerCount == 0)
                {
                    itemsWithNoWorkers.Add(item + 1);
                }
                else if (workerCount < 3)
                {
                    itemsWithInsufficientWorkers.Add(item + 1);
                }
            }

            if (itemsWithNoWorkers.Count > 0)
            {
                _validationMessages.Add(
                    $"WARNING: {itemsWithNoWorkers.Count} items have no annotations at all. They are still "
                    + "inferred, but their posterior is the prior and the reported label is arbitrary:");
                foreach (int item in itemsWithNoWorkers)
                {
                    _validationMessages.Add($"  - Item {item}");
                }
            }

            if (itemsWithInsufficientWorkers.Count > 0)
            {
                _validationMessages.Add($"WARNING: {itemsWithInsufficientWorkers.Count} items have fewer than 3 workers:");
                foreach (int item in itemsWithInsufficientWorkers)
                {
                    _validationMessages.Add($"  - Item {item}");
                }
            }

            var workersWithNoAnnotations = new List<int>();
            for (int worker = 0; worker < _numWorkers; worker++)
            {
                bool annotated = false;
                for (int item = 0; item < _numItems && !annotated; item++)
                {
                    annotated = _dataList[item][worker] != -1;
                }

                if (!annotated)
                {
                    workersWithNoAnnotations.Add(worker + 1);
                }
            }

            if (workersWithNoAnnotations.Count > 0)
            {
                _validationMessages.Add(
                    $"WARNING: {workersWithNoAnnotations.Count} workers annotated nothing. Their competence "
                    + "stays at the prior and carries no information:");
                foreach (int worker in workersWithNoAnnotations)
                {
                    _validationMessages.Add($"  - Worker {worker}");
                }
            }
        }

        /// <summary>
        /// Verifies that every integer in [0, numCategories-1] appears at least once in the data.
        /// A gap (e.g. labels {0, 2} with no label 1) would silently create a phantom category
        /// and skew inference, so it is treated as an error.
        /// </summary>
        private void ValidateLabelRange()
        {
            if (_dataList == null)
            {
                return;
            }

            var observedLabels = new HashSet<int>();
            foreach (var row in _dataList)
                foreach (var val in row)
                    if (val != -1)
                        observedLabels.Add(val);

            var missingLabels = Enumerable.Range(0, _numCategories)
                .Where(label => !observedLabels.Contains(label))
                .ToList();

            if (missingLabels.Count > 0)
            {
                var observed = string.Join(", ", observedLabels.OrderBy(x => x));
                var missing = string.Join(", ", missingLabels);
                throw new InvalidOperationException(
                    $"Label(s) {missing} are absent from the data but fall within the expected range [0, {_numCategories - 1}]. " +
                    $"Labels must form a contiguous range starting from 0. Observed labels: {observed}.");
            }
        }

        /// <summary>
        /// Disposes the underlying stream reader.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _streamReader?.Dispose();
                _disposed = true;
            }
        }
    }
}
