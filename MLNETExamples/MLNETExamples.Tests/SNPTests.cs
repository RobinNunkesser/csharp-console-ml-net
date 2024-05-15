using Microsoft.ML;
using MLNETExamples.ConsoleApp;

namespace MLNETExamples.Tests;

public class SNPTests
{
    [SetUp]
    public void Setup()
    {
    }
    
    private float CalculateMCR(string modelPath, string dataPath, string labelColumnName)
    {
        var mlContext = new MLContext();

        var data = SNP.LoadIDataViewFromFile(mlContext, dataPath, SNP.RetrainSeparatorChar, SNP.RetrainHasHeader, SNP.RetrainAllowQuoting);
        var mlModel = mlContext.Model.Load(modelPath, out var _);
        return SNP.CalculateMCR(data, mlModel,labelColumnName);
    }

    [Test]
    public void CalculateMCR()
    {
        var mcr = CalculateMCR(SNP.MLNetModelPath,
            SNP.RetrainFilePath, @"cl");
        Assert.That(mcr, Is.InRange(0.0, 0.1));
    } 
}
