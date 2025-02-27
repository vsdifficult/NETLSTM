using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Keras.PreProcessing.Text; 

namespace SentimentAnalysisLSTM.Data
{
    public class DataLoader
    {
        public (List<string> texts, List<int> labels) LoadData(string filePath)
        {
            var texts = new List<string>();
            var labels = new List<int>();

            foreach (var line in File.ReadLines(filePath).Skip(1)) // Пропускаем заголовок
            {
                var parts = line.Split(',');
                texts.Add(parts[0].Trim('"'));
                labels.Add(int.Parse(parts[1]));
            }

            return (texts, labels);
        }

        public void SaveTokenizer(Tokenizer tokenizer, string filePath)
        {
            using (var writer = new StreamWriter(filePath))
            {
                writer.WriteLine(tokenizer);
            }
        }

        public Tokenizer LoadTokenizer(string filePath)
        {
            var json = File.ReadAllText(filePath);
            return Tokenizer.FromJson(json);
        } 

        public int[][] PadSequences(Keras.Utils.Sequence sequences, int maxLength)

        {

            return sequences.Select(seq =>

            {

                if (seq.Length < maxLength)

                {

                    var padded = new int[maxLength];

                    Array.Copy(seq, padded, seq.Length);

                    return padded;

                }

                return seq.Take(maxLength).ToArray();

            }).ToArray();

        }
    }
}