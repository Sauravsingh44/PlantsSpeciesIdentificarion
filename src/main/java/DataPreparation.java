import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class DataPreparation {
    public static void main(String[] args) throws IOException {
        String datasetPath = "src/main/resources/flowers"; // Main flower dataset directory
        String trainDir = "./train";
        String testDir = "./test";

        // Print current working directory for debugging
        System.out.println("Current working directory: " + new File(".").getAbsolutePath());

        // Ensure train and test directories exist
        Files.createDirectories(Paths.get(trainDir));
        Files.createDirectories(Paths.get(testDir));

        File datasetFolder = new File(datasetPath);

        // Check if the dataset directory exists and list class folders
        if (!datasetFolder.exists()) {
            System.err.println("Dataset folder does not exist: " + datasetPath);
            return;
        }

        File[] classFolders = datasetFolder.listFiles();
        if (classFolders == null || classFolders.length == 0) {
            System.err.println("No class folders found in the dataset path.");
            return;
        }

        // Iterate through each class folder
        for (File classFolder : classFolders) {
            if (classFolder.isDirectory()) {
                System.out.println("Processing class folder: " + classFolder.getName());

                File[] files = classFolder.listFiles((dir, name) -> {
                    // Add an extra check to ensure only files (images) are listed, ignoring directories
                    return new File(dir, name).isFile() && (name.endsWith(".jpg") || name.endsWith(".png"));
                });

                if (files == null || files.length == 0) {
                    System.err.println("No image files found in class folder: " + classFolder.getName());
                    continue; // Skip if no files are found
                }

                // Shuffle and split files for training and testing
                List<File> fileList = Arrays.asList(files);
                Collections.shuffle(fileList); // Shuffle the files to randomize

                int trainSize = (int) (fileList.size() * 0.8); // 80% for training

                // Create subdirectories for each class inside train and test directories
                Files.createDirectories(Paths.get(trainDir, classFolder.getName()));
                Files.createDirectories(Paths.get(testDir, classFolder.getName()));

                // Copy 80% to train folder
                for (int i = 0; i < trainSize; i++) {
                    File file = fileList.get(i);
                    Files.copy(file.toPath(), Paths.get(trainDir, classFolder.getName(), file.getName()));
                }

                // Copy 20% to test folder
                for (int i = trainSize; i < fileList.size(); i++) {
                    File file = fileList.get(i);
                    Files.copy(file.toPath(), Paths.get(testDir, classFolder.getName(), file.getName()));
                }

                System.out.println("Finished processing class folder: " + classFolder.getName());
            }
        }
    }
}
