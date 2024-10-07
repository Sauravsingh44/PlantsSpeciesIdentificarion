import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;

public class FlowerClassifier {
    private static final String[] CLASS_NAMES = {"bougainvillea", "daisy", "frangipani", "hibiscus", "rose", "sunflower", "zinnia"};

    public static void main(String[] args) {
        // Example: 150x150 pixel images flattened to 22500-length arrays
        int inputSize = 150 * 150; // Flattened image size
        int hiddenSize1 = 128;       // Hidden layer size
        int hiddenSize2= 128;
        int outputSize = CLASS_NAMES.length; // Number of flower classes

        NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSize1,hiddenSize2, outputSize); // Using the NeuralNetwork class

        // Load flower images and labels into arrays
        double[][] trainData = loadFlowerImages("./train/");
        double[][] trainLabels = loadFlowerLabels("./train/");

        // Log the sizes of the training data and labels
        System.out.println("Training Data Size: " + trainData.length);
        System.out.println("Training Labels Size: " + trainLabels.length);

        // Ensure the training data and labels have the same number of samples
        if (trainData.length != trainLabels.length) {
            System.err.println("Mismatch between training data and labels. Exiting...");
            return;
        }

        // Train the neural network for 500 epochs
        nn.train(trainData, trainLabels, 300); // Use trainLabels directly

        // Test the network with test data
        double[][] testData = loadFlowerImages("./test/");
        double[][] testLabels = loadFlowerLabels("./test/");

        // Log the sizes of the test data and labels
        System.out.println("Test Data Size: " + testData.length);
        System.out.println("Test Labels Size: " + testLabels.length);

        int correctPredictions = 0;
        for (int i = 0; i < testData.length; i++) {
            int predictedClass = nn.predict(testData[i]);
            int actualClass = getLabelIndex(testLabels[i]);  // Convert one-hot to class index
            if (predictedClass == actualClass) {
                correctPredictions++;
            }
        }

        System.out.println("Accuracy: " + (correctPredictions * 100.0 / testData.length) + "%");

        // Example input image classification
//        String inputImagePath = "path/to/your/input/image.jpg"; // Replace with your image path
//        int predictedClass = predictFlowerClass(nn, inputImagePath);
//        System.out.println("Predicted class for the input image: " + CLASS_NAMES[predictedClass]);
    }

    // Method to load flower images and flatten them into arrays
    private static double[][] loadFlowerImages(String path) {
        List<double[]> images = new ArrayList<>();
        File folder = new File(path);
        File[] classFolders = folder.listFiles(File::isDirectory); // Get subdirectories (flower classes)

        if (classFolders != null) {
            for (File classFolder : classFolders) {
                String className = classFolder.getName(); // Get class name from folder name
                Integer classIndex = Arrays.asList(CLASS_NAMES).indexOf(className);
                File[] files = classFolder.listFiles((dir, name) -> name.endsWith(".jpg") || name.endsWith(".png"));

                if (files != null) {
                    for (File file : files) {
                        try {
                            BufferedImage img = ImageIO.read(file);
                            BufferedImage resizedImg = new BufferedImage(150, 150, BufferedImage.TYPE_INT_RGB);
                            Graphics2D g = resizedImg.createGraphics();
                            g.drawImage(img, 0, 0, 150, 150, null);
                            g.dispose();

                            // Convert image to double array (flattened)
                            double[] pixelValues = new double[150 * 150];
                            for (int i = 0; i < 150; i++) {
                                for (int j = 0; j < 150; j++) {
                                    int rgb = resizedImg.getRGB(j, i);
                                    // Normalize pixel values to [0, 1]
                                    pixelValues[i * 150 + j] = ((rgb >> 16) & 0xFF) / 255.0; // Red channel
                                }
                            }
                            images.add(pixelValues);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }
        return images.toArray(new double[0][]);
    }

    // Method to load flower labels based on folder names and convert to one-hot encoding
    private static double[][] loadFlowerLabels(String path) {
        List<double[]> labels = new ArrayList<>();

        // Map class names to indices
        Map<String, Integer> classMap = new HashMap<>();
        for (int i = 0; i < CLASS_NAMES.length; i++) {
            classMap.put(CLASS_NAMES[i], i);
        }

        // Get class folders and generate one-hot encoded labels
        File folder = new File(path);
        File[] classFolders = folder.listFiles(File::isDirectory); // Get subdirectories (flower classes)

        if (classFolders != null) {
            for (File classFolder : classFolders) {
                String className = classFolder.getName(); // Get class name from folder name
                Integer classIndex = classMap.get(className);
                File[] imageFiles = classFolder.listFiles((dir, name) -> name.endsWith(".jpg") || name.endsWith(".png"));
                if (imageFiles != null) {
                    for (File imageFile : imageFiles) {
                        double[] oneHotLabel = new double[CLASS_NAMES.length];
                        oneHotLabel[classIndex] = 1.0; // Set the index for the class to 1.0
                        labels.add(oneHotLabel);
                    }
                }
            }
        }

        // Convert list to array
        return labels.toArray(new double[0][]);
    }

    // Helper method to convert one-hot encoded label into index
    private static int getLabelIndex(double[] oneHotLabel) {
        for (int i = 0; i < oneHotLabel.length; i++) {
            if (oneHotLabel[i] == 1.0) return i;
        }
        return -1;  // Error case
    }

    // Method to predict the class of a single input image
    private static int predictFlowerClass(NeuralNetwork nn, String imagePath) {
        try {
            BufferedImage img = ImageIO.read(new File(imagePath));
            BufferedImage resizedImg = new BufferedImage(150, 150, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = resizedImg.createGraphics();
            g.drawImage(img, 0, 0, 150, 150, null);
            g.dispose();

            // Convert image to double array (flattened)
            double[] pixelValues = new double[150 * 150];
            for (int i = 0; i < 150; i++) {
                for (int j = 0; j < 150; j++) {
                    int rgb = resizedImg.getRGB(j, i);
                    // Normalize pixel values to [0, 1]
                    pixelValues[i * 150 + j] = ((rgb >> 16) & 0xFF) / 255.0; // Red channel
                }
            }

            return nn.predict(pixelValues); // Predict the class
        } catch (IOException e) {
            e.printStackTrace();
            return -1; // Return an invalid index in case of error
        }
    }
}
