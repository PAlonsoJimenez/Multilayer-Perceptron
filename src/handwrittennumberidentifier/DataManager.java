package handwrittennumberidentifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Scanner;

/**
 *
 * @author Pablo Alonso
 */
public class DataManager {
    
    public static double[][] loadTrainData() {
        Path path = Paths.get("train-images.idx3-ubyte");
        return loadData(path, 60000);
    }

    public static double[][] loadTestData() {
        Path path = Paths.get("t10k-images.idx3-ubyte");
        return loadData(path, 10000);
    }
    
    public static double[] loadTrainLabels() {
        Path path = Paths.get("train-labels.idx1-ubyte");
        return loadLabels(path, 60000);
    }

    public static double[] loadTestLabels() {
        Path path = Paths.get("t10k-labels.idx1-ubyte");
        return loadLabels(path, 10000);
    }
    
    public static boolean savePercepron(MultilayerPerceptron perceptron, String perceptronName, int testDataRightGuessed, int trainingDataRigthGuessed){
        for (String name : getPerceptronNames()) {
            if(perceptronName.equals(name)) return false;
        }
        try {
            savePerceptronData(perceptron, perceptronName, testDataRightGuessed, trainingDataRigthGuessed);
            return true;
        } catch (IOException ex) {
            System.out.println(ex);
        }
        return false;
    }
    
    /**
     * This method throw an error if the name passed as argument is not found in
     * the database index file.
     * @param perceptronName The name of the perceptron saved in the DataBase
     * @return the perceptron saved in the DataBase.
     */
    public static MultilayerPerceptron loadPerceptron(String perceptronName){
        boolean exist = false;
        for (String name : getPerceptronNames()) {
            if (perceptronName.equals(name)) {
                exist = true;
                break;
            }
        }
        if(!exist){
            throw new RuntimeException("Perceptron not found in the PerceptronIndex file");
        }
        return loadPerceptronData(perceptronName);
    }
    
    private static double[][] loadData(Path path, int numberOfImage){
        double[][] dataSet = new double[numberOfImage][784];
        try {
            byte[] images = Files.readAllBytes(path);
            int pointer = 16;
            //Scaling the input data from 0 to 1; 0 = 0, 255 = 1
            for (double[] image : dataSet) {
                for (int i = 0; i < image.length; i++) {
                    Byte pixelByte = images[pointer];
                    double pixel = pixelByte.doubleValue();
                    if (pixel < 0) {
                        pixel = 256 + pixel;
                    }
                    image[i] = pixel / 255;
                    pointer++;
                }
            }
            return dataSet;
        } catch (IOException ex) {
            System.out.println("PROBLEM WITH ONE DATA SET FILE");
            System.out.println(ex);
            System.exit(1);
        }
        return dataSet;
    }
    
    private static double[] loadLabels(Path path, int numberOfLabels){
        double[] labelsSet = new double[numberOfLabels];
        try {
            byte[] labels = Files.readAllBytes(path);
            int pointer = 8;
            //Scaling the input data from 0 to 0.9; 0 = 0, 9 = 0.9
            for (int i = 0; i < labelsSet.length; i++) {
                Byte labelByte = labels[pointer];
                double label = labelByte.doubleValue();
                labelsSet[i] = label / 10;
                pointer++;
            }
        } catch (IOException ex) {
            System.out.println("PROBLEM WITH ONE LABEL SET FILE");
            System.out.println(ex);
            System.exit(1);
        }
        return labelsSet;
    }

    private static ArrayList<String> getPerceptronNames() {
        try {
            File index = new File("PerceptronIndex");
            ArrayList<String> perceptronNames = new ArrayList<>();
            if (!index.exists()){
                index.createNewFile();
            } else {
                try (Scanner input = new Scanner(new BufferedReader(new FileReader(index)))) {
                    while (input.hasNextLine()) {
                        String line = input.nextLine();
                        if (line.isEmpty()) continue;
                        perceptronNames.add(line.trim());
                    }
                }
            }
            return perceptronNames;
        } catch (IOException ex) {
            System.out.println(ex);
        }
        return null;
    }
    
    private static void savePerceptronData(MultilayerPerceptron perceptron, String perceptronName, int testDataRightGuessed, int trainingDataRigthGuessed) throws IOException {
        //Index File
        File perceptronIndex = new File("PerceptronIndex");
        if(!perceptronIndex.exists()) perceptronIndex.createNewFile();
        PrintStream perceptronIndexline = new PrintStream(new FileOutputStream(perceptronIndex, true));
        perceptronIndexline.append(perceptronName + "\n");
        
        //Data File
        File newPerceptronData = new File(perceptronName);
        newPerceptronData.createNewFile();
        PrintStream line = new PrintStream(new FileOutputStream(newPerceptronData, true));
        String neuronsPerLayer = "NeuronsPerLayer:";
        for (int size : perceptron.getAllLayersSize()) {
            neuronsPerLayer += " " + size;
        }
        
        line.append("PerceptronID: " + perceptronName + "\n");
        line.append(neuronsPerLayer + "\n");
        line.append("AlphaConstant: " + perceptron.getAlpha() + "\n");
        line.append("TestDataRightGuessed: " + testDataRightGuessed + "\n");
        line.append("TrainingDataRigthGuessed: " + trainingDataRigthGuessed + "\n");
        line.append("Weights:" + "\n");
        String weightsValues;
        for (double[] weights : perceptron.getWeights()) {
            weightsValues = "";
            for (double weight : weights) {
                weightsValues += weight + " ";
            }
            line.append(weightsValues.trim() + "\n");
        }
        line.append("EndWeights" + "\n");
        
        line.append("Thresholds:" + "\n");
        String ThresholdsValues;
        for (double[] thresholds : perceptron.getThresholds()) {
            ThresholdsValues = "";
            for (double threshold : thresholds) {
                ThresholdsValues += threshold + " ";
            }
            line.append(ThresholdsValues.trim() + "\n");
        }
        line.append("EndThresholds" + "\n");
    }

    private static MultilayerPerceptron loadPerceptronData(String perceptronName) {
        try {
            File perceptronData = new File(perceptronName);
            if (!perceptronData.exists()) return null;
            try (Scanner input = new Scanner(new BufferedReader(new FileReader(perceptronData)))) {
                boolean loadingWeights = false;
                boolean loadingThresholds = false;
                double alpha = 0;
                int[] layerSizes = new int[0];
                LinkedList<double[]> weights = new LinkedList<>();
                LinkedList<double[]> thresholds = new LinkedList<>();
                while (input.hasNextLine()) {
                    String line = input.nextLine();
                    if (line.isEmpty()) {
                        continue;
                    }
                    String[] keyValueWords = line.split(" ");
                    if (loadingWeights) {
                        if (keyValueWords[0].equalsIgnoreCase("EndWeights")) {
                            loadingWeights = false;
                        } else {
                            double[] weightLayer = new double[keyValueWords.length];
                            for (int i = 0; i < keyValueWords.length; i++) {
                                weightLayer[i] = Double.parseDouble(keyValueWords[i]);
                            }
                            weights.add(weightLayer);
                        }
                    } else if (loadingThresholds) {
                        if (keyValueWords[0].equalsIgnoreCase("EndThresholds")) {
                            loadingThresholds = false;
                        } else {
                            double[] thresholdLayer = new double[keyValueWords.length];
                            for (int i = 0; i < keyValueWords.length; i++) {
                                thresholdLayer[i] = Double.parseDouble(keyValueWords[i]);
                            }
                            thresholds.add(thresholdLayer);
                        }
                    } else {
                        if (keyValueWords[0].equalsIgnoreCase("NeuronsPerLayer:")) {
                            layerSizes = new int[keyValueWords.length - 1];
                            for (int i = 1; i < keyValueWords.length; i++) {
                                layerSizes[i - 1] = Integer.parseInt(keyValueWords[i]);
                            }
                        }
                        if (keyValueWords[0].equalsIgnoreCase("AlphaConstant:")) {
                            alpha = Double.parseDouble(keyValueWords[1]);
                        }
                        if (keyValueWords[0].equalsIgnoreCase("Weights:")) {
                            loadingWeights = true;
                        }
                        if (keyValueWords[0].equalsIgnoreCase("Thresholds:")) {
                            loadingThresholds = true;
                        }
                    }
                }
                int[] hiddenLayersSizes = new int[layerSizes.length - 2];
                for (int i = 1; i < layerSizes.length - 1; i++) {
                    hiddenLayersSizes[i - 1] = layerSizes[i];
                }
                return new MultilayerPerceptron(layerSizes[0], layerSizes[layerSizes.length - 1], alpha, hiddenLayersSizes, weights, thresholds);
            }

        } catch (IOException ex) {
            System.out.println(ex);
        }
        return null;
    }
    
}
