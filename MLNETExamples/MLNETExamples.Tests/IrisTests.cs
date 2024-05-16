using Microsoft.ML;
using MLNETExamples.ConsoleApp;

namespace MLNETExamples.Tests;

public class IrisTests
{
    [Test] 
    public void RetrainModel()
    {
       Iris.Train("Iris.mlnet");     
    }
    
    [Test]
    public void CalculateMCR()
    {
        var mlContext = new MLContext();

        var data = Iris.LoadIDataViewFromFile(mlContext, Iris.RetrainFilePath, Iris.RetrainSeparatorChar, Iris.RetrainHasHeader, Iris.RetrainAllowQuoting);
        var mlModel = mlContext.Model.Load(Iris.MLNetModelPath, out var _);
        var mcr = Iris.CalculateMCR(data, mlModel,"Label");
        Assert.That(mcr,Is.InRange(0.1,0.11));
    }
}