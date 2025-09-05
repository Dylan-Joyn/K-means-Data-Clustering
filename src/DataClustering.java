import java.io.*;
import java.util.*;

public class DataClustering {

    public static void main(String[] args) throws FileNotFoundException {
        //Command Line Arguments
        String filename = args[0];
        int numClusters = Integer.parseInt(args[1]);
        int maxIterations = Integer.parseInt(args[2]);
        double threshold = Double.parseDouble(args[3]);
        int numRuns = Integer.parseInt(args[4]);

        //Read Data on File
        double[][] data = readData(filename);

        //Get K cluster starting centers
        double[][] centers = selectStartingCenters(data, numClusters);

        //Print the centers
        printCenters(centers);
    }

    //Reads the data and returns it as a 2D array
    public static double[][] readData(String filename) throws FileNotFoundException {
        Scanner fileScanner = new Scanner(new File(filename));


        int numPoints = fileScanner.nextInt();
        int dimension = fileScanner.nextInt();

        //Array to store all the data points
        double[][] data = new double[numPoints][dimension];

        //Reads each data point
        for (int i=0; i<numPoints; i++){
            for (int j=0; j < dimension; j++){
                data[i][j] = fileScanner.nextDouble();
            }
        }
        fileScanner.close();
        return data;
    }

    //Grab K clusters centers
    public static double[][] selectStartingCenters(double[][] data, int k){
        int numPoints = data.length;
        int dimension = data[0].length;

        //Array to store centers
        double[][] centers = new double[k][dimension];

        //Keeps track of points we selected
        Set<Integer> selectedIndexes = new HashSet<>();

        Random rand = new Random();

        //Select random K points
        int centersSelected = 0;
        while (centersSelected < k){
            int randomIndex = rand.nextInt(numPoints);

            //Determines if we have selected the point yet, if not adds to center
            if(!selectedIndexes.contains(randomIndex)){

                //Puts datapoint to centers array
                for (int i=0; i<dimension; i++){
                    centers[centersSelected][i] = data[randomIndex][i];
                }
                selectedIndexes.add(randomIndex);
                centersSelected++;
            }
        }
        return centers;
    }



    //Prints centers to output
    public static void printCenters(double[][] centers) {
        for (int i = 0; i < centers.length; i++) {
            for (int j = 0; j < centers[i].length; j++) {
                System.out.print(centers[i][j]);
                if (j < centers[i].length - 1) {
                    System.out.print(" ");
                }
            }
            System.out.println();
        }
    }
}