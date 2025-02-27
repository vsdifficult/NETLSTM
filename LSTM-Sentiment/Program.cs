using System;
using SentimentAnalysisLSTM.Models;
using SentimentAnalysisLSTM.Predictions;

namespace SentimentAnalysisLSTM
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainer = new Trainer();
            trainer.TrainModel("Data/train.csv");

            var predictor = new Predictor("sentiment_model.h5", "tokenizer .json");
            Console.WriteLine("Enter a text for sentiment analysis:");
            var text = Console.ReadLine();
            if (string.IsNullOrEmpty(text))
            {
                Console.WriteLine("No text provided for sentiment analysis.");
                return;
            }
            var sentiment = predictor.PredictSentiment(text);

            Console.WriteLine($"The sentiment is: {sentiment}");
        }
    }
}
