using Microsoft.ML;
using MLNETExamples.ConsoleApp;

namespace MLNETExamples.Tests;

public class TaxiTests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void TestFirstPrediction()
    {
        var sampleData = new Taxi.ModelInput()
        {
            Vendor_id = @"CMT",
            Rate_code = 1F,
            Passenger_count = 1F,
            Trip_time_in_secs = 1271F,
            Trip_distance = 3.8F,
            Payment_type = @"CRD",
        };
        var predictionResult = Taxi.Predict(sampleData);
        Assert.That(predictionResult.Score, Is.InRange(16.96,16.97));
    }

    [Test]
    public void PlotRSquared()
    {
        var mlContext = new MLContext();

        var data = Taxi.LoadIDataViewFromFile(mlContext, Taxi.RetrainFilePath, Taxi.RetrainSeparatorChar, Taxi.RetrainHasHeader, Taxi.RetrainAllowQuoting);
        var mlModel = mlContext.Model.Load(Taxi.MLNetModelPath, out var _);
        Taxi.PlotRSquaredValues(data, mlModel,@"fare_amount","./");
    }
}