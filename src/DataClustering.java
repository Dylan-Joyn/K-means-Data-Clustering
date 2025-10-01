/**
 * Author: James Joyner
 * Course: Data Clustering
 * Phase 1 & 2 implementation.
 * Oracle Java Coding Conventions:
 * https://www.oracle.com/java/technologies/javase/codeconventions-contents.html
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class DataClustering {

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length < 5) {
            System.out.println("Usage: java DataClustering <filename> <k> <maxIterations> <threshold> <numRuns>");
            return;
        }

        // command line arguments
        String filename = args[0];
        int numClusters = Integer.parseInt(args[1]);
        int maxIterations = Integer.parseInt(args[2]);
        double threshold = Double.parseDouble(args[3]);
        int numRuns = Integer.parseInt(args[4]);

        // read dataset
        double[][] data = readData(filename);

        double bestSSE = Double.MAX_VALUE;
        int bestRun = -1;

        // run K-Means multiple times
        for (int run = 1; run <= numRuns; run++) {
            System.out.println("Run " + run + "\n-----");

            // initialize centers
            double[][] centers = selectStartingCenters(data, numClusters);
            int[] assignments = new int[data.length];

            double prevSSE = Double.MAX_VALUE;
            double sse = 0.0;

            for (int iter = 1; iter <= maxIterations; iter++) {
                // assign points to nearest cluster
                for (int i = 0; i < data.length; i++) {
                    assignments[i] = nearestCenter(data[i], centers);
                }

                // recalculate cluster centers
                centers = recomputeCenters(data, assignments, numClusters);

                // calculate SSE
                sse = computeSSE(data, assignments, centers);
                System.out.printf("Iteration %d: SSE = %.6f%n", iter, sse);

                // check convergence
                if (Math.abs(prevSSE - sse) < threshold) {
                    break;
                }
                prevSSE = sse;
            }

            // track best run
            if (sse < bestSSE) {
                bestSSE = sse;
                bestRun = run;
            }
            System.out.println();
        }

        System.out.printf("Best Run: %d: SSE = %.6f%n", bestRun, bestSSE);
    }

    // reads dataset
    private static double[][] readData(String filename) throws FileNotFoundException {
        Scanner fileScanner = new Scanner(new File(filename));
        int numPoints = fileScanner.nextInt();
        int dimension = fileScanner.nextInt();

        double[][] data = new double[numPoints][dimension];
        for (int i = 0; i < numPoints; i++) {
            for (int j = 0; j < dimension; j++) {
                data[i][j] = fileScanner.nextDouble();
            }
        }
        fileScanner.close();
        return data;
    }

    // select K random centers
    private static double[][] selectStartingCenters(double[][] data, int k) {
        int numPoints = data.length;
        int dim = data[0].length;
        double[][] centers = new double[k][dim];

        Random rand = new Random();
        Set<Integer> selected = new HashSet<>();
        int chosen = 0;

        while (chosen < k) {
            int idx = rand.nextInt(numPoints);
            if (!selected.contains(idx)) {
                centers[chosen] = Arrays.copyOf(data[idx], dim);
                selected.add(idx);
                chosen++;
            }
        }
        return centers;
    }

    // nearest cluster for a point
    private static int nearestCenter(double[] point, double[][] centers) {
        double minDist = Double.MAX_VALUE;
        int cluster = -1;
        for (int i = 0; i < centers.length; i++) {
            double dist = squaredDistance(point, centers[i]);
            if (dist < minDist) {
                minDist = dist;
                cluster = i;
            }
        }
        return cluster;
    }

    // recompute cluster centers
    private static double[][] recomputeCenters(double[][] data, int[] assignments, int k) {
        int dim = data[0].length;
        double[][] newCenters = new double[k][dim];
        int[] counts = new int[k];

        for (int i = 0; i < data.length; i++) {
            int cluster = assignments[i];
            counts[cluster]++;
            for (int d = 0; d < dim; d++) {
                newCenters[cluster][d] += data[i][d];
            }
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < dim; d++) {
                    newCenters[c][d] /= counts[c];
                }
            }
        }
        return newCenters;
    }

    // fixed compute SSE
    private static double computeSSE(double[][] data, int[] assignments, double[][] centers) {
        double sse = 0.0;
        for (int i = 0; i < data.length; i++) {
            int cluster = assignments[i];
            sse += squaredDistance(data[i], centers[cluster]);
        }
        return sse;
    }

    // squared Euclidean distance
    private static double squaredDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
