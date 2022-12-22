using System;
using System.Diagnostics;
using System.Linq;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        double[][] inputSignal;
        double[][,] weights;
        double[][] errors;
        double learningConst = 0.025;

        Stopwatch stopWatch = new Stopwatch();
        Random rand = new Random();
        int minWeight = -1;
        int maxWeight = 1;
        public StudentNetwork(int[] structure)
        {

            inputSignal = new double[structure.Length][];
            errors = new double[structure.Length][];

            // инициализация начальных значений
            for (int i = 0; i < structure.Length; i++)
            {
                errors[i] = new double[structure[i]];
                inputSignal[i] = new double[structure[i] + 1];
                inputSignal[i][structure[i]] = 0; 
            }

            weights = new double[structure.Length - 1][,]; 

            // заполняем матрицу весов случайными значениями
            for (int n = 0; n < structure.Length - 1; n++)
            {
                int rowsCount = structure[n] + 1;
                int columnsCount = structure[n + 1];

                weights[n] = new double[rowsCount, columnsCount];

                for (int i = 0; i < rowsCount; i++)
                {
                    for (int j = 0; j < columnsCount; j++)
                    {
                        double randWeight = minWeight + rand.NextDouble() * (maxWeight - minWeight);
                        weights[n][i, j] = randWeight;
                    }    
                }     
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iteration = 1;
            bool f = true;

            if (sample.error != null)
                f = sample.EstimatedError() > acceptableError;
            while (f)
            {
                // прогон по сети
                Run(sample.input);

                // дельта(ошибка) для последнего слоя
                for (var i = 0; i < sample.Output.Length; i++)
                {
                    double currentSignal = inputSignal[errors.Length - 1][i];
                    double expected = sample.Output[i];
                    errors[errors.Length - 1][i] = currentSignal * (1 - currentSignal) * (expected - currentSignal);
                }
                // перенос ошибки вглубь сети
                for (int i = errors.Length - 2; i >= 1; i--)
                {
                    for (int j = 0; j < errors[i].Length; j++)
                    {
                        double yi = inputSignal[i][j];
                        double errorPart = yi * (1 - yi);

                        double sum = 0;
                        for (int k = 0; k < errors[i + 1].Length; k++)
                        {
                            sum += errors[i + 1][k] * weights[i][j, k];
                        }
                        errors[i][j] = errorPart * sum;
                    }
                }

                // корректировка весов
                for (int n = 0; n < weights.Length; n++)
                {
                    for (int i = 0; i < weights[n].GetLength(0); i++)
                    {
                        for (int j = 0; j < weights[n].GetLength(1); j++)
                        {
                            double dw = learningConst * errors[n + 1][j] * inputSignal[n][i];
                            weights[n][i, j] += dw;
                        }
                    }
                }

                iteration++;

                if (sample.error == null)
                    return iteration;
                f = sample.EstimatedError() > acceptableError;
            }

            return iteration;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            // Конструируем массивы входов и выходов
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            // Группируем массивы из samplesSet в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            int currentEpoch = 0;
            double samplesLooked = 0;
            double samplesCount = inputs.Length * epochsCount;
            double error = double.PositiveInfinity;

            stopWatch.Restart();

            while (currentEpoch++ < epochsCount && error > acceptableError)
            {
                error = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    Train(samplesSet[i], acceptableError, parallel);
                    error += EstimatedErrorFromOutput(outputs[i]);
                    samplesLooked++;
                }
                // среднее значение ошибки
                error /= inputs.Length;
                OnTrainProgress(samplesLooked / samplesCount, error, stopWatch.Elapsed);
            }

            OnTrainProgress(1, error, stopWatch.Elapsed);
            stopWatch.Stop();
            return error;
        }

        protected override double[] Compute(double[] input)
        {
            Run(input);
            return inputSignal[inputSignal.Length - 1].Take(inputSignal.Last().Length - 1).ToArray();
        }

        private void Run(double[] input)
        {
            for (int j = 0; j < input.Length; j++)
                inputSignal[0][j] = input[j];

            for (int i = 1; i < inputSignal.GetLength(0); i++)
                Activate(inputSignal[i - 1], inputSignal[i], weights[i - 1]);
        }

        private double EstimatedErrorFromOutput(double[] output)
        {
            double result = 0;

            for (int i = 0; i < output.Length; i++)
            {
                double yi = inputSignal[inputSignal.Length - 1][i];
                result += Math.Pow(output[i] - yi, 2);
            }
          
            //result /= output.Length;

            return result;
        }

        private void Activate(double[] prevLayer, double[] layer, double[,] weights)
        {
            int rowsCount = weights.GetLength(0);
            int colCount = weights.GetLength(1);

            for (int i = 0; i < colCount; i++)
            {
                double sum = 0;

                for (int j = 0; j < rowsCount; j++)
                    sum += prevLayer[j] * weights[j, i];

                layer[i] = ActivateFunction(sum);
            }
        }

        // сигмоида
        private double ActivateFunction(double x) => 1.0 / (Math.Exp(-x) + 1);
    }
}