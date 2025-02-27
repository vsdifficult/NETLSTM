using System;
using Tensorflow;
using Keras;
using Keras.Layers; 
using Keras.Models; 

namespace SentimentAnalysisLSTM.Models
{
    public class LSTMModel
    {
        public Sequential CreateModel(int vocabSize, int embeddingDim = 100, int maxLength = 100)
        {
            var model = new Sequential();
            model.Add(new Embedding(input_dim: vocabSize, output_dim: embeddingDim, input_length: maxLength));
            model.Add(new LSTM(units: 100));
            model.Add(new Dense(units: 1, activation: "sigmoid"));
            model.Compile(optimizer: "adam", loss: "binary_crossentropy", metrics: new[] { "accuracy" });
            return model;
        }
    }
}