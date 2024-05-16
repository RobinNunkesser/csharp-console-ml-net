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

        for (int i = 0; i < predictedValues.Length; i++)
        {
            var label = -1;
            switch (trueValues[i])
            {
                case "Iris-setosa":
                    label=2;
                    break;
                case "Iris-versicolor":
                    label = 1;
                    break;
                case "Iris-virginica":
                    label = 3;
                    break;
            }
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