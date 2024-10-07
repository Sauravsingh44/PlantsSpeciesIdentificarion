import java.util.Random;

public class NeuralNetwork {
    private int inputSize, hiddenSize, outputSize;
    private double[][] weightsInputHidden, weightsHiddenOutput;
    private double[] hiddenBias, outputBias;
    private double learningRate = 0.01;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Initialize weights and biases randomly
        weightsInputHidden = new double[hiddenSize][inputSize];
        weightsHiddenOutput = new double[outputSize][hiddenSize];
        hiddenBias = new double[hiddenSize];
        outputBias = new double[outputSize];

        Random random = new Random();
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightsInputHidden[i][j] = random.nextGaussian();
            }
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsHiddenOutput[i][j] = random.nextGaussian();
            }
        }
    }

    // Activation function: Sigmoid
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Derivative of sigmoid (for backpropagation)
    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    // Forward propagation
    public double[] forward(double[] input) {
        double[] hiddenLayer = new double[hiddenSize];
        double[] outputLayer = new double[outputSize];

        // Input to hidden
        for (int i = 0; i < hiddenSize; i++) {
            hiddenLayer[i] = 0;
            for (int j = 0; j < inputSize; j++) {
                hiddenLayer[i] += input[j] * weightsInputHidden[i][j];
            }
            hiddenLayer[i] += hiddenBias[i];
            hiddenLayer[i] = sigmoid(hiddenLayer[i]);
        }

        // Hidden to output
        for (int i = 0; i < outputSize; i++) {
            outputLayer[i] = 0;
            for (int j = 0; j < hiddenSize; j++) {
                outputLayer[i] += hiddenLayer[j] * weightsHiddenOutput[i][j];
            }
            outputLayer[i] += outputBias[i];
            outputLayer[i] = sigmoid(outputLayer[i]);
        }

        return outputLayer;
    }

    // Backpropagation
    public void backward(double[] input, double[] targetOutput, double[] output) {
        double[] hiddenLayer = new double[hiddenSize];
        double[] outputLayer = new double[outputSize];
        double[] outputError = new double[outputSize];
        double[] hiddenError = new double[hiddenSize];

        // Calculate output error
        for (int i = 0; i < outputSize; i++) {
            outputError[i] = targetOutput[i] - output[i];
        }

        // Backpropagate to hidden layer
        for (int i = 0; i < hiddenSize; i++) {
            hiddenError[i] = 0;
            for (int j = 0; j < outputSize; j++) {
                hiddenError[i] += outputError[j] * weightsHiddenOutput[j][i];
            }
            hiddenError[i] *= sigmoidDerivative(hiddenLayer[i]);
        }

        // Update weights and biases between hidden and output layer
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsHiddenOutput[i][j] += learningRate * outputError[i] * hiddenLayer[j];
            }
            outputBias[i] += learningRate * outputError[i];
        }

        // Update weights and biases between input and hidden layer
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightsInputHidden[i][j] += learningRate * hiddenError[i] * input[j];
            }
            hiddenBias[i] += learningRate * hiddenError[i];
        }
    }

    // Train the neural network
    public void train(double[][] trainData, double[][] trainLabels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainData.length; i++) {
                double[] output = forward(trainData[i]);
                backward(trainData[i], trainLabels[i], output);
            }
            if (epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + " completed");
            }
        }
    }

    // Predict
    public int predict(double[] input) {
        double[] output = forward(input);
        int predictedClass = 0;
        double maxVal = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxVal) {
                maxVal = output[i];
                predictedClass = i;
            }
        }
        return predictedClass;
    }
}
