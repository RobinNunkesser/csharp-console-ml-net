using Microsoft.ML;
using MLNETExamples.ConsoleApp;

namespace MLNETExamples.Tests;

public class RestaurantTests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Retrain()
    {
        Restaurant.Train("restaurantRetrain.mlnet");
    }

    [Test]
    public void CalculateMCR()
    {
        var mlContext = new MLContext();

        var data = Restaurant.LoadIDataViewFromFile(mlContext, Restaurant.RetrainFilePath, Restaurant.RetrainSeparatorChar, Restaurant.RetrainHasHeader, Restaurant.RetrainAllowQuoting);
        var mlModel = mlContext.Model.Load(Restaurant.MLNetModelPath, out var _);
        var mcr = Restaurant.CalculateMCR(data, mlModel,@"will_wait");
        Assert.That(mcr, Is.InRange(0.4, 0.6));
    }
}