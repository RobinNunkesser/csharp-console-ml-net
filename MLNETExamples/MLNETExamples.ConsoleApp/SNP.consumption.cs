// This file was auto-generated by ML.NET Model Builder.
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
namespace MLNETExamples.ConsoleApp
{
    public partial class SNP
    {
        /// <summary>
        /// model input class for SNP.
        /// </summary>
        #region model input class
        public class ModelInput
        {
            [LoadColumn(0)]
            [ColumnName(@"SNP1")]
            public float SNP1 { get; set; }

            [LoadColumn(1)]
            [ColumnName(@"SNP2")]
            public float SNP2 { get; set; }

            [LoadColumn(2)]
            [ColumnName(@"SNP3")]
            public float SNP3 { get; set; }

            [LoadColumn(3)]
            [ColumnName(@"SNP4")]
            public float SNP4 { get; set; }

            [LoadColumn(4)]
            [ColumnName(@"SNP5")]
            public float SNP5 { get; set; }

            [LoadColumn(5)]
            [ColumnName(@"SNP6")]
            public float SNP6 { get; set; }

            [LoadColumn(6)]
            [ColumnName(@"SNP7")]
            public float SNP7 { get; set; }

            [LoadColumn(7)]
            [ColumnName(@"SNP8")]
            public float SNP8 { get; set; }

            [LoadColumn(8)]
            [ColumnName(@"SNP9")]
            public float SNP9 { get; set; }

            [LoadColumn(9)]
            [ColumnName(@"SNP10")]
            public float SNP10 { get; set; }

            [LoadColumn(10)]
            [ColumnName(@"SNP11")]
            public float SNP11 { get; set; }

            [LoadColumn(11)]
            [ColumnName(@"SNP12")]
            public float SNP12 { get; set; }

            [LoadColumn(12)]
            [ColumnName(@"SNP13")]
            public float SNP13 { get; set; }

            [LoadColumn(13)]
            [ColumnName(@"SNP14")]
            public float SNP14 { get; set; }

            [LoadColumn(14)]
            [ColumnName(@"SNP15")]
            public float SNP15 { get; set; }

            [LoadColumn(15)]
            [ColumnName(@"cl")]
            public float Cl { get; set; }

        }

        #endregion

        /// <summary>
        /// model output class for SNP.
        /// </summary>
        #region model output class
        public class ModelOutput
        {
            [ColumnName(@"SNP1")]
            public float SNP1 { get; set; }

            [ColumnName(@"SNP2")]
            public float SNP2 { get; set; }

            [ColumnName(@"SNP3")]
            public float SNP3 { get; set; }

            [ColumnName(@"SNP4")]
            public float SNP4 { get; set; }

            [ColumnName(@"SNP5")]
            public float SNP5 { get; set; }

            [ColumnName(@"SNP6")]
            public float SNP6 { get; set; }

            [ColumnName(@"SNP7")]
            public float SNP7 { get; set; }

            [ColumnName(@"SNP8")]
            public float SNP8 { get; set; }

            [ColumnName(@"SNP9")]
            public float SNP9 { get; set; }

            [ColumnName(@"SNP10")]
            public float SNP10 { get; set; }

            [ColumnName(@"SNP11")]
            public float SNP11 { get; set; }

            [ColumnName(@"SNP12")]
            public float SNP12 { get; set; }

            [ColumnName(@"SNP13")]
            public float SNP13 { get; set; }

            [ColumnName(@"SNP14")]
            public float SNP14 { get; set; }

            [ColumnName(@"SNP15")]
            public float SNP15 { get; set; }

            [ColumnName(@"cl")]
            public uint Cl { get; set; }

            [ColumnName(@"Features")]
            public float[] Features { get; set; }

            [ColumnName(@"PredictedLabel")]
            public float PredictedLabel { get; set; }

            [ColumnName(@"Score")]
            public float[] Score { get; set; }

        }

        #endregion

        public static string MLNetModelPath = Path.GetFullPath("SNP.mlnet");

        public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);


        private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        }

        /// <summary>
        /// Use this method to predict scores for all possible labels.
        /// </summary>
        /// <param name="input">model input.</param>
        /// <returns><seealso cref=" ModelOutput"/></returns>
        public static IOrderedEnumerable<KeyValuePair<string, float>> PredictAllLabels(ModelInput input)
        {
            var predEngine = PredictEngine.Value;
            var result = predEngine.Predict(input);
            return GetSortedScoresWithLabels(result);
        }

        /// <summary>
        /// Map the unlabeled result score array to the predicted label names.
        /// </summary>
        /// <param name="result">Prediction to get the labeled scores from.</param>
        /// <returns>Ordered list of label and score.</returns>
        /// <exception cref="Exception"></exception>
        public static IOrderedEnumerable<KeyValuePair<string, float>> GetSortedScoresWithLabels(ModelOutput result)
        {
            var unlabeledScores = result.Score;
            var labelNames = GetLabels(result);

            Dictionary<string, float> labledScores = new Dictionary<string, float>();
            for (int i = 0; i < labelNames.Count(); i++)
            {
                // Map the names to the predicted result score array
                var labelName = labelNames.ElementAt(i);
                labledScores.Add(labelName.ToString(), unlabeledScores[i]);
            }

            return labledScores.OrderByDescending(c => c.Value);
        }

        /// <summary>
        /// Get the ordered label names.
        /// </summary>
        /// <param name="result">Predicted result to get the labels from.</param>
        /// <returns>List of labels.</returns>
        /// <exception cref="Exception"></exception>
        private static IEnumerable<string> GetLabels(ModelOutput result)
        {
            var schema = PredictEngine.Value.OutputSchema;

            var labelColumn = schema.GetColumnOrNull("cl");
            if (labelColumn == null)
            {
                throw new Exception("cl column not found. Make sure the name searched for matches the name in the schema.");
            }

            // Key values contains an ordered array of the possible labels. This allows us to map the results to the correct label value.
            var keyNames = new VBuffer<float>();
            labelColumn.Value.GetKeyValues(ref keyNames);
            return keyNames.DenseValues().Select(x => x.ToString());
        }

        /// <summary>
        /// Use this method to predict on <see cref="ModelInput"/>.
        /// </summary>
        /// <param name="input">model input.</param>
        /// <returns><seealso cref=" ModelOutput"/></returns>
        public static ModelOutput Predict(ModelInput input)
        {
            var predEngine = PredictEngine.Value;
            return predEngine.Predict(input);
        }
    }
}