using System.IO;
using Microsoft.ML.Data;

namespace MLNETExamples.ConsoleApp;


public partial class Iris
{
    /// <summary>
    /// model input class for Iris.
    /// </summary>
    #region model input class
    public class ModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"SepalLength")]
        public float SepalLength { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"SepalWidth")]
        public float SepalWidth { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"PetalLength")]
        public float PetalLength { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"PetalWidth")]
        public float PetalWidth { get; set; }
        
        [LoadColumn(4)]
        [ColumnName(@"Label")]
        public string Label { get; set; }
    }

    #endregion

    /// <summary>
    /// model output class for Iris.
    /// </summary>
    #region model output class
    public class ModelOutput
    {
        [ColumnName(@"PredictedLabel")]
        public uint PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[]? Distances { get; set; }

    }

    #endregion
    
    public static string MLNetModelPath = Path.GetFullPath("Iris.mlnet");
}
