using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNETExamples.ConsoleApp;

public partial class Iris
{
    public static float CalculateMCR(IDataView trainData, ITransformer model,
        string labelColumnName)
    {
        // Use the model to make batch predictions on training data
        var testResults = model.Transform(trainData);

        // Get the actual values from the dataset
        var trueValues = testResults.GetColumn<string>(labelColumnName).ToArray();

        // Get the predicted values from the test results
        var predictedValues = testResults.GetColumn<uint>("PredictedLabel").ToArray(); 
        var predictionScores = testResults.GetColumn<float[]>("Score");

        foreach (var score in predictionScores)
        {
            Console.WriteLine($"Distances: {string.Join(" ", score ?? Array.Empty<float>())}");            
        }
        
        
        
        var mcr = 0F;
        
        var setosaLabel = (int)predictedValues.Take(50).GroupBy(x => x).OrderByDescending(g => g.Count()).First().Key;
        var versicolorLabel = (int)predictedValues.Skip(50).Take(50).GroupBy(x => x).OrderByDescending(g => g.Count()).First().Key;
        var virginicaLabel = (int)predictedValues.Skip(100).Take(50).GroupBy(x => x).OrderByDescending(g => g.Count()).First().Key;
        
        for (var i = 0; i < predictedValues.Length; i++)
        {
            var label = trueValues[i] switch
            {
                "Iris-setosa" => setosaLabel,
                "Iris-versicolor" => versicolorLabel,
                "Iris-virginica" => virginicaLabel,
                _ => -1
            };
            {
                
            }
            if (predictedValues[i] != label)
            {
                mcr++;
            }
                    
        }

        mcr /= predictedValues.Length;
        return mcr;
        
    }
}