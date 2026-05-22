namespace MACE
{
    /// <summary>
    /// Reads and parses CSV files containing crowdsourcing annotation data.
    /// The CSV format expects a header row with worker names, followed by rows of annotation data.
    /// Missing annotations are represented by empty cells.
    /// 
    /// Data Quality Requirements:
    /// - Each work item must be seen by at least 3 different workers
    /// - Workers cannot see the same work item more than once (duplicates are handled)
    /// </summary>
    public class CsvReader : IDisposable
    {
        private readonly StreamReader _streamReader;
        private List<int[]>? _dataList;
        private int _numWorkers;
        private int _numItems;
        private int _numCategories;
        private bool _disposed = false;
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
        /// <exception cref="InvalidOperationException">Thrown when the file format is invalid or reading fails.</exception>
        public void Read()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(CsvReader));
            }

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
        /// <returns>A two-dimensional array where data[item][worker] gives the annotation for that item-worker pair, or -1 for missing annotations.</returns>
        /// <exception cref="InvalidOperationException">Thrown when Read() has not been called or when no data is available.</exception>
        public int[][] GetData()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(CsvReader));
            }

            if (_dataList == null)
            {
                throw new InvalidOperationException("Read() must be called before GetData().");
            }

            var data = new int[_numItems][];
            for (int i = 0; i < _numItems; i++)
            {
                data[i] = new int[_numWorkers];
                int[] item = _dataList[i];
                for (int j = 0; j < _numWorkers; j++)
                {
                    data[i][j] = item[j];
                }
            }
            return data;
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
        /// Checks that each item has been annotated by at least 3 workers and emits
        /// a warning for any that fall below that threshold.
        /// </summary>
        private void CheckWorkerCoverage()
        {
            if (_dataList == null)
            {
                return;
            }

            _validationMessages.Clear();

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

                if (workerCount < 3)
                {
                    itemsWithInsufficientWorkers.Add(item + 1);
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
