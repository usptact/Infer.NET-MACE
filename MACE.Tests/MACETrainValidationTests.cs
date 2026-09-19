using MACE;

namespace MACE.Tests
{
    public class MACETrainValidationTests
    {
        [Theory]
        [InlineData(0, 2, 2, 50)]
        [InlineData(-1, 2, 2, 50)]
        [InlineData(2, 0, 2, 50)]
        [InlineData(2, 2, 0, 50)]
        [InlineData(2, 2, 2, 0)]
        [InlineData(2, 2, 2, -5)]
        public void Constructor_NonPositiveArguments_Throw(int workers, int items, int categories, int iterations)
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new MACETrain(workers, items, categories, iterations));
        }

        [Fact]
        public void InferModelData_NullAnnotations_Throws()
        {
            var trainer = new MACETrain(3, 2, 2);

            Assert.Throws<ArgumentNullException>(
                () => trainer.InferModelData(null!, TestSupport.UniformPriors(3, 2)));
        }

        [Fact]
        public void InferModelData_NullPriors_Throws()
        {
            var trainer = new MACETrain(3, 2, 2);
            var annotations = new SparseAnnotations(
                new[] { new[] { 0, 1 }, new[] { 1, 2 } },
                new[] { new[] { 0, 1 }, new[] { 1, 0 } });

            Assert.Throws<ArgumentNullException>(() => trainer.InferModelData(annotations, null!));
        }

        [Fact]
        public void InferModelData_ItemCountMismatch_Throws()
        {
            var trainer = new MACETrain(3, 5, 2);
            var annotations = new SparseAnnotations(
                new[] { new[] { 0, 1 } },
                new[] { new[] { 0, 1 } });

            var ex = Assert.Throws<ArgumentException>(
                () => trainer.InferModelData(annotations, TestSupport.UniformPriors(3, 2)));
            Assert.Contains("items but model expects", ex.Message);
        }
    }
}
