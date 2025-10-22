/**
 * Author: James Joyner
 * Course: Data Clustering
 * Phase 1–3 Implementation (Random Selection + Random Partition)
 * Oracle Java Coding Conventions:
 * https://www.oracle.com/java/technologies/javase/codeconventions-contents.html
 */

import java.io.*;
import java.util.*;

public class DataClustering {

    public static void main(String[] args) throws FileNotFoundException {

        if (args.length < 6) {
            System.out.println("Usage: java DataClustering <filename> <k> <maxIterations> <threshold> <numRuns> <initMethod>");
            System.out.println("Initialization methods: random, partition");
            return;
        }

        String filename = args[0];
        int numClusters = Integer.parseInt(args[1]);
        int maxIterations = Integer.parseInt(args[2]);
        double threshold = Double.parseDouble(args[3]);
        int numRuns = Integer.parseInt(args[4]);
        String initMethod = args[5].toLowerCase();

        double[][] rawData = readData(filename);
        double[][] data = minMaxNormalize(rawData);

        double bestSSE = Double.MAX_VALUE;
        int bestRun = -1;

        // ***** ADDED FOR EXCEL OUTPUT *****
        double bestInitialSSE = Double.MAX_VALUE;
        double bestFinalSSE = Double.MAX_VALUE;
        int bestIterationCount = 0;

        // multiple runs
        for (int run = 1; run <= numRuns; run++) {
            System.out.println("\nRun " + run + " (" + initMethod + " initialization)");
            System.out.println("-----------------------------------");

            double[][] centers;
            if (initMethod.equals("partition")) {
                centers = randomPartitionInit(data, numClusters);
            } else {
                centers = selectStartingCenters(data, numClusters);
            }

            int[] assignments = new int[data.length];

            // Compute initial SSE before first iteration
            for (int i = 0; i < data.length; i++) {
                assignments[i] = nearestCenter(data[i], centers);
            }
            double initialSSE = computeSSE(data, assignments, centers);
            System.out.printf("Initial SSE: %.6f%n", initialSSE);

            // ***** ADDED FOR EXCEL OUTPUT *****
            if (initialSSE < bestInitialSSE) {
                bestInitialSSE = initialSSE;
            }

            double prevSSE = Double.MAX_VALUE;
            double sse = 0.0;
            int iter;

            for (iter = 1; iter <= maxIterations; iter++) {

                for (int i = 0; i < data.length; i++) {
                    assignments[i] = nearestCenter(data[i], centers);
                }

                sse = computeSSE(data, assignments, centers);
                System.out.printf("Iteration %d: SSE = %.6f%n", iter, sse);

                if ((prevSSE - sse) / prevSSE < threshold) {
                    break;
                }

                prevSSE = sse;
                centers = recomputeCenters(data, assignments, numClusters);
            }

            System.out.printf("Final SSE after %d iterations: %.6f%n", iter, sse);

            // ***** ADDED FOR EXCEL OUTPUT *****
            if (sse < bestFinalSSE) {
                bestFinalSSE = sse;
                bestIterationCount = iter;
            }

            if (sse < bestSSE) {
                bestSSE = sse;
                bestRun = run;
            }
        }

        System.out.printf("%nBest Run: %d, SSE = %.6f%n", bestRun, bestSSE);

        // ***** ADDED FOR EXCEL OUTPUT (print summary) *****
        System.out.println("\n=== SUMMARY FOR EXCEL ===");
        System.out.printf("Best Initial SSE: %.6f%n", bestInitialSSE);
        System.out.printf("Best Final SSE: %.6f%n", bestFinalSSE);
        System.out.printf("Best Iteration Count: %d%n", bestIterationCount);

    }

    // === helper functions remain unchanged ===

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

    private static double[][] minMaxNormalize(double[][] data) {
        int numPoints = data.length;
        int dim = data[0].length;
        double[][] normalized = new double[numPoints][dim];

        double[] minVals = new double[dim];
        double[] maxVals = new double[dim];
        Arrays.fill(minVals, Double.MAX_VALUE);
        Arrays.fill(maxVals, -Double.MAX_VALUE);

        for (int i = 0; i < numPoints; i++) {
            for (int d = 0; d < dim; d++) {
                if (data[i][d] < minVals[d]) minVals[d] = data[i][d];
                if (data[i][d] > maxVals[d]) maxVals[d] = data[i][d];
            }
        }

        for (int i = 0; i < numPoints; i++) {
            for (int d = 0; d < dim; d++) {
                if (maxVals[d] - minVals[d] == 0) {
                    normalized[i][d] = 0.0;
                } else {
                    normalized[i][d] = (data[i][d] - minVals[d]) / (maxVals[d] - minVals[d]);
                }
            }
        }
        return normalized;
    }

    private static double[][] selectStartingCenters(double[][] data, int k) {
        int numPoints = data.length;
        int dim = data[0].length;
        double[][] centers = new double[k][dim];
        Random rand = new Random();
        Set<Integer> selected = new HashSet<>();

        for (int chosen = 0; chosen < k; ) {
            int idx = rand.nextInt(numPoints);
            if (!selected.contains(idx)) {
                centers[chosen] = Arrays.copyOf(data[idx], dim);
                selected.add(idx);
                chosen++;
            }
        }
        return centers;
    }

    private static double[][] randomPartitionInit(double[][] data, int k) {
        int numPoints = data.length;
        int dim = data[0].length;
        double[][] centers = new double[k][dim];
        int[] counts = new int[k];
        Random rand = new Random();

        for (int i = 0; i < numPoints; i++) {
            int cluster = rand.nextInt(k);
            counts[cluster]++;
            for (int d = 0; d < dim; d++) {
                centers[cluster][d] += data[i][d];
            }
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < dim; d++) {
                    centers[c][d] /= counts[c];
                }
            }
        }
        return centers;
    }

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

    private static double computeSSE(double[][] data, int[] assignments, double[][] centers) {
        double sse = 0.0;
        for (int i = 0; i < data.length; i++) {
            int cluster = assignments[i];
            sse += squaredDistance(data[i], centers[cluster]);
        }
        return sse;
    }

    private static double squaredDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
