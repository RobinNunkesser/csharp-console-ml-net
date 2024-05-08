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

    private float CalculateMCR(string modelPath, string dataPath, string labelColumnName)
    {
        var mlContext = new MLContext();

        var data = Restaurant.LoadIDataViewFromFile(mlContext, dataPath, Restaurant.RetrainSeparatorChar, Restaurant.RetrainHasHeader, Restaurant.RetrainAllowQuoting);
        var mlModel = mlContext.Model.Load(modelPath, out var _);
        return Restaurant.CalculateMCR(data, mlModel,labelColumnName);
    }

    [Test]
    public void CalculateMCR()
    {
        var mcr = CalculateMCR(Restaurant.MLNetModelPath,
            Restaurant.RetrainFilePath, @"will_wait");
        Assert.That(mcr, Is.InRange(0.4, 0.6));
    }
    
    [Test]
    public void CalculateMCRCompleteModelAIMAData()
    {
        var mcr = CalculateMCR(Restaurant.MLNetCompleteModelPath,
            Restaurant.RetrainFilePath, @"will_wait");
        Assert.That(mcr, Is.InRange(0.0, 0.1));
    }
    
    [Test]
    public void CalculateMCR5xModelAIMAData()
    {
        var mcr = CalculateMCR(Restaurant.MLNet5xModelPath,
            Restaurant.RetrainFilePath, @"will_wait");
        Assert.That(mcr, Is.InRange(0.0, 0.1));
    }

    
    [Test]
    public void CalculateMCRCompleteModelCompleteData()
    {
        var mcr = CalculateMCR(Restaurant.MLNetCompleteModelPath,
            Restaurant.RetrainCompleteFilePath, @"will_wait");
        Assert.That(mcr, Is.InRange(0.0, 0.1));
    }
    
    [Test]
    public void CalculateMCR5xModelCompleteData()
    {
        var mcr = CalculateMCR(Restaurant.MLNet5xModelPath,
            Restaurant.RetrainCompleteFilePath, @"will_wait");
        Assert.That(mcr, Is.InRange(0.5, 0.6));
    }
}