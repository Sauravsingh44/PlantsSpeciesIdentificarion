import java.util.Random;

public class NeuralNetwork {
    private int inputSize, hiddenSize1, hiddenSize2,hiddenSize3, outputSize;
    private double[][] weightsInputHidden1, weightsHidden1Hidden2,weightsHidden2Hidden3, weightsHiddenOutput;
    private double[] hiddenBias1, hiddenBias2,hiddenBias3, outputBias;
    private double learningRate = 0.02;

    public NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2,int hiddenSize3, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize1 = hiddenSize1;
        this.hiddenSize2 = hiddenSize2;
        this.hiddenSize3=hiddenSize3;
        this.outputSize = outputSize;

        // Initialize weights and biases using Xavier initialization
        weightsInputHidden1 = new double[hiddenSize1][inputSize];
        weightsHidden1Hidden2 = new double[hiddenSize2][hiddenSize1];
        weightsHidden2Hidden3=new double[hiddenSize3][hiddenSize2];
        weightsHiddenOutput = new double[outputSize][hiddenSize2];
        hiddenBias1 = new double[hiddenSize1];
        hiddenBias2 = new double[hiddenSize2];
        hiddenBias3=new double[hiddenSize3];
        outputBias = new double[outputSize];

        Random random = new Random();
        initializeWeights(weightsInputHidden1, inputSize, hiddenSize1, random);
        initializeWeights(weightsHidden1Hidden2, hiddenSize1, hiddenSize2, random);
        initializeWeights(weightsHidden2Hidden3,hiddenSize2,hiddenSize3,random);
        initializeWeights(weightsHiddenOutput, hiddenSize2, outputSize, random);
    }

    private void initializeWeights(double[][] weights, int inputSize, int outputSize, Random random) {
        double stddev = Math.sqrt(2.0 / (inputSize + outputSize));
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = random.nextGaussian() * stddev;
            }
        }
    }

    // Activation function: ReLU
    private double relu(double x) {
        return Math.max(0, x);
    }

    // Derivative of ReLU (for backpropagation)
    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    // Activation function: Softmax
    private double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double val : x) {
            if (val > max) max = val;
        }
        double sum = 0;
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.exp(x[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }
        return result;
    }

    // Forward propagation
    public double[] forward(double[] input) {
        double[] hiddenLayer1 = new double[hiddenSize1];
        double[] hiddenLayer2 = new double[hiddenSize2];
        double[] outputLayer = new double[outputSize];

        // Input to first hidden layer
        for (int i = 0; i < hiddenSize1; i++) {
            hiddenLayer1[i] = 0;
            for (int j = 0; j < inputSize; j++) {
                hiddenLayer1[i] += input[j] * weightsInputHidden1[i][j];
            }
            hiddenLayer1[i] += hiddenBias1[i];
            hiddenLayer1[i] = relu(hiddenLayer1[i]);
        }

        // First hidden layer to second hidden layer
        for (int i = 0; i < hiddenSize2; i++) {
            hiddenLayer2[i] = 0;
            for (int j = 0; j < hiddenSize1; j++) {
                hiddenLayer2[i] += hiddenLayer1[j] * weightsHidden1Hidden2[i][j];
            }
            hiddenLayer2[i] += hiddenBias2[i];
            hiddenLayer2[i] = relu(hiddenLayer2[i]);
        }

        // Second hidden layer to output layer
        for (int i = 0; i < outputSize; i++) {
            outputLayer[i] = 0;
            for (int j = 0; j < hiddenSize2; j++) {
                outputLayer[i] += hiddenLayer2[j] * weightsHiddenOutput[i][j];
            }
            outputLayer[i] += outputBias[i];
        }

        return softmax(outputLayer);
    }

    // Backpropagation
    public void backward(double[] input, double[] targetOutput, double[] output) {
        double[] hiddenLayer1 = new double[hiddenSize1];
        double[] hiddenLayer2 = new double[hiddenSize2];
        double[] outputError = new double[outputSize];
        double[] hiddenError2 = new double[hiddenSize2];
        double[] hiddenError1 = new double[hiddenSize1];

        // Calculate output error
        for (int i = 0; i < outputSize; i++) {
            outputError[i] = targetOutput[i] - output[i];
        }

        // Backpropagate to second hidden layer
        for (int i = 0; i < hiddenSize2; i++) {
            hiddenError2[i] = 0;
            for (int j = 0; j < outputSize; j++) {
                hiddenError2[i] += outputError[j] * weightsHiddenOutput[j][i];
            }
            hiddenError2[i] *= reluDerivative(hiddenLayer2[i]);
        }

        // Backpropagate to first hidden layer
        for (int i = 0; i < hiddenSize1; i++) {
            hiddenError1[i] = 0;
            for (int j = 0; j < hiddenSize2; j++) {
                hiddenError1[i] += hiddenError2[j] * weightsHidden1Hidden2[j][i];
            }
            hiddenError1[i] *= reluDerivative(hiddenLayer1[i]);
        }

        // Update weights and biases between second hidden and output layer
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize2; j++) {
                weightsHiddenOutput[i][j] += learningRate * outputError[i] * hiddenLayer2[j];
            }
            outputBias[i] += learningRate * outputError[i];
        }

        // Update weights and biases between first hidden and second hidden layer
        for (int i = 0; i < hiddenSize2; i++) {
            for (int j = 0; j < hiddenSize1; j++) {
                weightsHidden1Hidden2[i][j] += learningRate * hiddenError2[i] * hiddenLayer1[j];
            }
            hiddenBias2[i] += learningRate * hiddenError2[i];
        }

        // Update weights and biases between input and first hidden layer
        for (int i = 0; i < hiddenSize1; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightsInputHidden1[i][j] += learningRate * hiddenError1[i] * input[j];
            }
            hiddenBias1[i] += learningRate * hiddenError1[i];
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
            printProgressBar(epoch + 1, epochs);
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
    private void printProgressBar(int current, int total) {
        int progressBarLength = 50; // Length of the progress bar
        int progress = (int) ((double) current / total * progressBarLength);

        StringBuilder progressBar = new StringBuilder("[");
        for (int i = 0; i < progressBarLength; i++) {
            if (i < progress) {
                progressBar.append("=");
            } else {
                progressBar.append(" ");
            }
        }
        progressBar.append("] ").append(current).append("/").append(total).append(" epochs");

        // Print the progress bar
        System.out.print("\r" + progressBar.toString());
        if (current == total) {
            System.out.println();
        }
    }
}