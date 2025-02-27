using System;
using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Layers;
using Keras.Optimizers;
using Keras.PreProcessing.Text;
using Keras.PreProcessing; 

namespace SentimentAnalysisLSTM.Models
{
    public class Trainer
    {
        public void TrainModel(string trainFile)
        {
            var dataLoader = new DataLoader();
            var (texts, labels) = dataLoader.LoadData(trainFile);

            // Токенизация
            var tokenizer = new Tokenizer(num_words: 10000);
            tokenizer.FitOnTexts(texts);
            var sequences = tokenizer.TextsToSequences(texts);
            var paddedSequences = sequences.(sequences, maxlen: 100);

            dataLoader.SaveTokenizer(tokenizer, "tokenizer.json");

            var model = new LSTMModel().CreateModel(vocabSize: 10000); 

            model.Fit(paddedSequences, labels, batch_size: 32, epochs: 10);
            model.Save("sentiment_model.h5");
            Console.WriteLine("Model saved as sentiment_model.h5");
        }
    }
}