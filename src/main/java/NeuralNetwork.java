import java.util.Random;

public class NeuralNetwork {
    private int inputSize, hiddenSize1, hiddenSize2, hiddenSize3, outputSize;
    private double[][] weightsInputHidden1, weightsHidden1Hidden2, weightsHidden2Hidden3, weightsHiddenOutput;
    private double[] hiddenBias1, hiddenBias2, hiddenBias3, outputBias;
    private double learningRate = 0.02;

    // Convolutional layer parameters
    private int numFilters;
    private int filterSize;
    private double[][][] filters;
    private double[] convBias;

    // Pooling layer parameters
    private int poolSize;

    public NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2, int hiddenSize3, int outputSize,
                         int numFilters, int filterSize, int poolSize) {
        this.inputSize = inputSize;
        this.hiddenSize1 = hiddenSize1;
        this.hiddenSize2 = hiddenSize2;
        this.hiddenSize3 = hiddenSize3;
        this.outputSize = outputSize;
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.poolSize = poolSize;

        // Initialize weights and biases using Xavier initialization
        weightsInputHidden1 = new double[hiddenSize1][inputSize];
        weightsHidden1Hidden2 = new double[hiddenSize2][hiddenSize1];
        weightsHidden2Hidden3 = new double[hiddenSize3][hiddenSize2];
        weightsHiddenOutput = new double[outputSize][hiddenSize2];
        hiddenBias1 = new double[hiddenSize1];
        hiddenBias2 = new double[hiddenSize2];
        hiddenBias3 = new double[hiddenSize3];
        outputBias = new double[outputSize];

        Random random = new Random();
        initializeWeights(weightsInputHidden1, inputSize, hiddenSize1, random);
        initializeWeights(weightsHidden1Hidden2, hiddenSize1, hiddenSize2, random);
        initializeWeights(weightsHidden2Hidden3, hiddenSize2, hiddenSize3, random);
        initializeWeights(weightsHiddenOutput, hiddenSize2, outputSize, random);

        // Initialize convolutional layer
        filters = new double[numFilters][filterSize][filterSize];
        convBias = new double[numFilters];
        for (int i = 0; i < numFilters; i++) {
            initializeWeights(filters[i], filterSize * filterSize, 1, random);
            convBias[i] = 0;
        }
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

    // Convolution operation
    private double[][] convLayer(double[][] input, double[][] filter, double bias) {
        int inputSize = input.length;
        int filterSize = filter.length;
        int outputSize = inputSize - filterSize + 1;
        double[][] output = new double[outputSize][outputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double sum = 0;
                for (int m = 0; m < filterSize; m++) {
                    for (int n = 0; n < filterSize; n++) {
                        sum += input[i + m][j + n] * filter[m][n];
                    }
                }
                output[i][j] = relu(sum + bias);
            }
        }
        return output;
    }

    // Pooling operation (max pooling)
    private double[][] poolLayer(double[][] input) {
        int inputSize = input.length;
        int outputSize = inputSize / poolSize;
        double[][] output = new double[outputSize][outputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int m = 0; m < poolSize; m++) {
                    for (int n = 0; n < poolSize; n++) {
                        max = Math.max(max, input[i * poolSize + m][j * poolSize + n]);
                    }
                }
                output[i][j] = max;
            }
        }
        return output;
    }

    // Forward propagation
    public double[] forward(double[] input) {
        // Reshape input for convolutional layer
        int inputDim = (int) Math.sqrt(input.length);
        double[][] reshapedInput = new double[inputDim][inputDim];
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < inputDim; j++) {
                reshapedInput[i][j] = input[i * inputDim + j];
            }
        }

        // Convolutional layer
        double[][] convOutput = new double[reshapedInput.length - filterSize + 1][reshapedInput.length - filterSize + 1];
        for (int i = 0; i < numFilters; i++) {
            double[][] convResult = convLayer(reshapedInput, filters[i], convBias[i]);
            for (int j = 0; j < convOutput.length; j++) {
                for (int k = 0; k < convOutput[j].length; k++) {
                    convOutput[j][k] += convResult[j][k];
                }
            }
        }

        // Pooling layer
        double[][] pooledOutput = poolLayer(convOutput);

        // Flatten the pooled output
        double[] flattenedPooledOutput = new double[pooledOutput.length * pooledOutput.length];
        for (int i = 0; i < pooledOutput.length; i++) {
            for (int j = 0; j < pooledOutput[i].length; j++) {
                flattenedPooledOutput[i * pooledOutput.length + j] = pooledOutput[i][j];
            }
        }

        // Input to first hidden layer
        double[] hiddenLayer1 = new double[hiddenSize1];
        for (int i = 0; i < hiddenSize1; i++) {
            hiddenLayer1[i] = 0;
            for (int j = 0; j < flattenedPooledOutput.length; j++) {
                hiddenLayer1[i] += flattenedPooledOutput[j] * weightsInputHidden1[i][j];
            }
            hiddenLayer1[i] += hiddenBias1[i];
            hiddenLayer1[i] = relu(hiddenLayer1[i]);
        }

        // First hidden layer to second hidden layer
        double[] hiddenLayer2 = new double[hiddenSize2];
        for (int i = 0; i < hiddenSize2; i++) {
            hiddenLayer2[i] = 0;
            for (int j = 0; j < hiddenSize1; j++) {
                hiddenLayer2[i] += hiddenLayer1[j] * weightsHidden1Hidden2[i][j];
            }
            hiddenLayer2[i] += hiddenBias2[i];
            hiddenLayer2[i] = relu(hiddenLayer2[i]);
        }

        // Second hidden layer to third hidden layer
        double[] hiddenLayer3 = new double[hiddenSize3];
        for (int i = 0; i < hiddenSize3; i++) {
            hiddenLayer3[i] = 0;
            for (int j = 0; j < hiddenSize2; j++) {
                hiddenLayer3[i] += hiddenLayer2[j] * weightsHidden2Hidden3[i][j];
            }
            hiddenLayer3[i] += hiddenBias3[i];
            hiddenLayer3[i] = relu(hiddenLayer3[i]);
        }

        // Third hidden layer to output layer
        double[] outputLayer = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputLayer[i] = 0;
            for (int j = 0; j < hiddenSize3; j++) {
                outputLayer[i] += hiddenLayer3[j] * weightsHiddenOutput[i][j];
            }
            outputLayer[i] += outputBias[i];
        }

        return softmax(outputLayer);
    }

    // Backpropagation
    public void backward(double[] input, double[] targetOutput, double[] output) {
        double[] hiddenLayer1 = new double[hiddenSize1];
        double[] hiddenLayer2 = new double[hiddenSize2];
        double[] hiddenLayer3 = new double[hiddenSize3];
        double[] outputError = new double[outputSize];
        double[] hiddenError3 = new double[hiddenSize3];
        double[] hiddenError2 = new double[hiddenSize2];
        double[] hiddenError1 = new double[hiddenSize1];

        // Calculate output error
        for (int i = 0; i < outputSize; i++) {
            outputError[i] = targetOutput[i] - output[i];
        }

        // Backpropagate to third hidden layer
        for (int i = 0; i < hiddenSize3; i++) {
            hiddenError3[i] = 0;
            for (int j = 0; j < outputSize; j++) {
                hiddenError3[i] += outputError[j] * weightsHiddenOutput[j][i];
            }
            hiddenError3[i] *= reluDerivative(hiddenLayer3[i]);
        }

        // Backpropagate to second hidden layer
        for (int i = 0; i < hiddenSize2; i++) {
            hiddenError2[i] = 0;
            for (int j = 0; j < hiddenSize3; j++) {
                hiddenError2[i] += hiddenError3[j] * weightsHidden2Hidden3[j][i];
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

        // Update weights and biases between third hidden and output layer
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize3; j++) {
                weightsHiddenOutput[i][j] += learningRate * outputError[i] * hiddenLayer3[j];
            }
            outputBias[i] += learningRate * outputError[i];
        }

        // Update weights and biases between second hidden and third hidden layer
        for (int i = 0; i < hiddenSize3; i++) {
            for (int j = 0; j < hiddenSize2; j++) {
                weightsHidden2Hidden3[i][j] += learningRate * hiddenError3[i] * hiddenLayer2[j];
            }
            hiddenBias3[i] += learningRate * hiddenError3[i];
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
