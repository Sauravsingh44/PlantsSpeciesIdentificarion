import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ND4J {
    public static void main(String[] args) {
        // Create two 2x2 matrices
        INDArray matrixA = Nd4j.create(new double[][]{
                {1, 2},
                {3, 4}
        });

        INDArray matrixB = Nd4j.create(new double[][]{
                {5, 6},
                {7, 8}
        });

        // Perform addition of the two matrices
        INDArray matrixAdd = matrixA.add(matrixB);
        System.out.println("Matrix Addition:\n" + matrixAdd);

        // Perform matrix multiplication
        INDArray matrixMul = matrixA.mmul(matrixB);
        System.out.println("Matrix Multiplication:\n" + matrixMul);
    }
}