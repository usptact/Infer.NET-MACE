using MACE;

namespace MACE.Tests
{
    public class CsvReaderTests
    {
        [Fact]
        public void Read_ParsesDimensionsFromHeaderAndRows()
        {
            using var csv = new TempCsv("w1,w2,w3", "0,1,0", "1,1,0");
            using var reader = new CsvReader(csv.Path);
            reader.Read();

            Assert.Equal(3, reader.GetNumWorkers());
            Assert.Equal(2, reader.GetNumItems());
        }

        /// <summary>
        /// GetNumCategories returns the category COUNT, not the highest label value.
        /// Callers pass it to the model unchanged; adding 1 would invent a phantom category.
        /// </summary>
        [Fact]
        public void GetNumCategories_ReturnsCountNotMaxLabelValue()
        {
            using var csv = new TempCsv("w1,w2,w3", "0,1,2", "2,1,0");
            using var reader = new CsvReader(csv.Path);
            reader.Read();

            Assert.Equal(3, reader.GetNumCategories());
        }

        [Fact]
        public void GetSparseData_KeepsOnlyObservedCellsAndStaysParallel()
        {
            using var csv = new TempCsv("w1,w2,w3,w4", "0,,1,", ",1,,0");
            using var reader = new CsvReader(csv.Path);
            reader.Read();

            var sparse = reader.GetSparseData();

            Assert.Equal(new[] { 0, 2 }, sparse.WorkerIndices[0]);
            Assert.Equal(new[] { 0, 1 }, sparse.Labels[0]);
            Assert.Equal(new[] { 1, 3 }, sparse.WorkerIndices[1]);
            Assert.Equal(new[] { 1, 0 }, sparse.Labels[1]);

            for (int item = 0; item < sparse.WorkerIndices.Length; item++)
            {
                Assert.Equal(sparse.WorkerIndices[item].Length, sparse.Labels[item].Length);
            }
        }

        [Fact]
        public void GetSparseData_BeforeRead_Throws()
        {
            using var csv = new TempCsv("w1,w2,w3", "0,1,0");
            using var reader = new CsvReader(csv.Path);

            Assert.Throws<InvalidOperationException>(() => reader.GetSparseData());
        }

        [Fact]
        public void Constructor_MissingFile_Throws()
        {
            Assert.Throws<FileNotFoundException>(
                () => new CsvReader(Path.Combine(Path.GetTempPath(), $"absent_{Guid.NewGuid():N}.csv")));
        }

        [Theory]
        [InlineData("")]
        [InlineData("   ")]
        public void Constructor_BlankFileName_Throws(string fileName)
        {
            Assert.Throws<ArgumentException>(() => new CsvReader(fileName));
        }

        [Fact]
        public void Read_RaggedRow_Throws()
        {
            using var csv = new TempCsv("w1,w2,w3", "0,1,0", "1,1");
            using var reader = new CsvReader(csv.Path);

            var ex = Assert.Throws<InvalidOperationException>(() => reader.Read());
            Assert.Contains("columns", ex.Message);
        }

        [Fact]
        public void Read_NegativeLabel_Throws()
        {
            using var csv = new TempCsv("w1,w2,w3", "0,-1,1");
            using var reader = new CsvReader(csv.Path);

            var ex = Assert.Throws<InvalidOperationException>(() => reader.Read());
            Assert.Contains("non-negative", ex.Message);
        }

        [Fact]
        public void Read_NonIntegerCell_Throws()
        {
            using var csv = new TempCsv("w1,w2,w3", "0,abc,1");
            using var reader = new CsvReader(csv.Path);

            Assert.Throws<InvalidOperationException>(() => reader.Read());
        }

        /// <summary>
        /// Labels {0, 2} with no 1 would silently create an unused category and skew inference,
        /// so a gap is an error rather than a warning.
        /// </summary>
        [Fact]
        public void Read_LabelGap_Throws()
        {
            using var csv = new TempCsv("w1,w2,w3", "0,2,0", "2,0,2");
            using var reader = new CsvReader(csv.Path);

            var ex = Assert.Throws<InvalidOperationException>(() => reader.Read());
            Assert.Contains("contiguous", ex.Message);
        }

        [Fact]
        public void Read_NoDataRows_Throws()
        {
            using var csv = new TempCsv("w1,w2,w3");
            using var reader = new CsvReader(csv.Path);

            Assert.Throws<InvalidOperationException>(() => reader.Read());
        }

        [Fact]
        public void Read_BlankLinesAreSkipped()
        {
            using var csv = new TempCsv("w1,w2,w3", "0,1,0", "", "1,1,0");
            using var reader = new CsvReader(csv.Path);
            reader.Read();

            Assert.Equal(2, reader.GetNumItems());
        }

        [Fact]
        public void Read_ItemBelowWorkerThreshold_WarnsWithoutThrowing()
        {
            using var csv = new TempCsv("w1,w2,w3,w4", "0,1,0,1", "0,,,", "1,1,0,0");
            using var reader = new CsvReader(csv.Path);
            reader.Read();

            var messages = reader.GetValidationMessages();
            Assert.Contains(messages, m => m.Contains("fewer than 3 workers"));
            Assert.Contains(messages, m => m.Contains("Item 2"));
        }

        /// <summary>
        /// An item nobody annotated is still inferred, but it is called out on its own rather than
        /// folded into the thin-coverage warning, because its reported label carries no information.
        /// </summary>
        [Fact]
        public void Read_ItemWithNoAnnotations_WarnsSeparately()
        {
            using var csv = new TempCsv("w1,w2,w3,w4", "0,1,0,1", ",,,", "1,1,0,0");
            using var reader = new CsvReader(csv.Path);
            reader.Read();

            var messages = reader.GetValidationMessages();
            Assert.Contains(messages, m => m.Contains("no annotations at all"));
            Assert.Contains(messages, m => m.Contains("Item 2"));
            Assert.DoesNotContain(messages, m => m.Contains("fewer than 3 workers"));
        }

        [Fact]
        public void Read_ItemWithNoAnnotations_IsStillReturnedAsAnEmptySlot()
        {
            using var csv = new TempCsv("w1,w2,w3,w4", "0,1,0,1", ",,,", "1,1,0,0");
            using var reader = new CsvReader(csv.Path);
            reader.Read();

            var sparse = reader.GetSparseData();

            Assert.Equal(3, reader.GetNumItems());
            Assert.Empty(sparse.WorkerIndices[1]);
            Assert.Empty(sparse.Labels[1]);
        }

        [Fact]
        public void Read_ZeroAndThinCoverage_AreReportedIndependently()
        {
            using var csv = new TempCsv("w1,w2,w3,w4", "0,1,0,1", ",,,", "1,,,");
            using var reader = new CsvReader(csv.Path);
            reader.Read();

            var messages = reader.GetValidationMessages();
            Assert.Contains(messages, m => m.Contains("no annotations at all"));
            Assert.Contains(messages, m => m.Contains("fewer than 3 workers"));
        }

        [Fact]
        public void Read_FullyCoveredData_ProducesNoWarnings()
        {
            using var csv = new TempCsv("w1,w2,w3", "0,1,0", "1,1,0");
            using var reader = new CsvReader(csv.Path);
            reader.Read();

            Assert.Empty(reader.GetValidationMessages());
        }
    }
}
