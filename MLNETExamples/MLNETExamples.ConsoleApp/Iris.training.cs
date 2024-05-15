using System.IO;
using Microsoft.ML;

namespace MLNETExamples.ConsoleApp;

public partial class Iris
{
    public const string RetrainFilePath = @"iris.data";
    public const char RetrainSeparatorChar = ',';
    public const bool RetrainHasHeader = false;
    public const bool RetrainAllowQuoting = false;

    /// <summary>
    /// Train a new model with the provided dataset.
    /// </summary>
    /// <param name="outputModelPath">File path for saving the model. Should be similar to "C:\YourPath\ModelName.mlnet"</param>
    /// <param name="inputDataFilePath">Path to the data file for training.</param>
    /// <param name="separatorChar">Separator character for delimited training file.</param>
    /// <param name="hasHeader">Boolean if training file has a header.</param>
    public static void Train(string outputModelPath,
        string inputDataFilePath = RetrainFilePath,
        char separatorChar = RetrainSeparatorChar,
        bool hasHeader = RetrainHasHeader,
        bool allowQuoting = RetrainAllowQuoting)
    {
        var mlContext = new MLContext();

        var data = LoadIDataViewFromFile(mlContext, inputDataFilePath,
            separatorChar, hasHeader, allowQuoting);
        var model = RetrainModel(mlContext, data);
        SaveModel(mlContext, model, data, outputModelPath);
    }

    /// <summary>
    /// Load an IDataView from a file path.
    /// </summary>
    /// <param name="mlContext">The common context for all ML.NET operations.</param>
    /// <param name="inputDataFilePath">Path to the data file for training.</param>
    /// <param name="separatorChar">Separator character for delimited training file.</param>
    /// <param name="hasHeader">Boolean if training file has a header.</param>
    /// <returns>IDataView with loaded training data.</returns>
    public static IDataView LoadIDataViewFromFile(MLContext mlContext,
        string inputDataFilePath, char separatorChar, bool hasHeader,
        bool allowQuoting)
    {
        return mlContext.Data.LoadFromTextFile<ModelInput>(inputDataFilePath,
            separatorChar, hasHeader, allowQuoting: allowQuoting);
    }


    /// <summary>
    /// Save a model at the specified path.
    /// </summary>
    /// <param name="mlContext">The common context for all ML.NET operations.</param>
    /// <param name="model">Model to save.</param>
    /// <param name="data">IDataView used to train the model.</param>
    /// <param name="modelSavePath">File path for saving the model. Should be similar to "C:\YourPath\ModelName.mlnet.</param>
    public static void SaveModel(MLContext mlContext, ITransformer model,
        IDataView data, string modelSavePath)
    {
        // Pull the data schema from the IDataView used for training the model
        DataViewSchema dataViewSchema = data.Schema;

        using (var fs = File.Create(modelSavePath))
        {
            mlContext.Model.Save(model, dataViewSchema, fs);
        }
    }


    /// <summary>
    /// Retrain model using the pipeline generated as part of the training process.
    /// </summary>
    /// <param name="mlContext"></param>
    /// <param name="trainData"></param>
    /// <returns></returns>
    public static ITransformer RetrainModel(MLContext mlContext,
        IDataView trainData)
    {
        var pipeline = BuildPipeline(mlContext);
        var model = pipeline.Fit(trainData);

        return model;
    }

    /// <summary>
    /// build the pipeline that is used from model builder. Use this function to retrain model.
    /// </summary>
    /// <param name="mlContext"></param>
    /// <returns></returns>
    public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
    {
        const string featuresColumnName = "Features";
        var pipeline = mlContext.Transforms
            .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
            .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));
        

        return pipeline;
    }
}

