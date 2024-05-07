using MLNETExamples.ConsoleApp;

namespace MLNETExamples.Tests;

public class Tests
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
}