using System;
using System.IO;
using Tensorflow;
using Tensorflow.Keras.Models; // Ensure this is available
using Keras.PreProcessing.Text; // Ensure this is available
using Tensorflow.Keras; // Add this line to include missing components
using Keras.Models; 

namespace SentimentAnalysisLSTM.Predictions
{
    public class Predictor
    {
        private readonly Model model;
        private readonly Tokenizer tokenizer;

        public Predictor(string modelPath, string tokenizerPath)
        {
            model = LoadModel(modelPath);
            tokenizer = LoadTokenizer(tokenizerPath);
        }

        private static Model LoadModel(string modelPath) // Ensure the method is static
        {
            return Model.LoadModel(modelPath);
        }

        private static Tokenizer LoadTokenizer(string tokenizerPath) // Ensure the method is static
        {
            var json = File.ReadAllText(tokenizerPath);
            return Tokenizer.FromJson(json);
        }

        public string PredictSentiment(string text)
        {
            if (text == null) throw new ArgumentNullException(nameof(text));

            // Токенизация и преобразование текста
            var sequence = tokenizer.TextsToSequences(new[] { text });
            var paddedSequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen: 100);

            var prediction = model .Predict(paddedSequence);
            return prediction[0][0] > 0.5 ? "Positive" : "Negative";
        }
    }
}
