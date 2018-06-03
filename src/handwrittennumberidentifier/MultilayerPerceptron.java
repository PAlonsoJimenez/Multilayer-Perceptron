package handwrittennumberidentifier;

import java.util.LinkedList;
import java.util.Random;

/**
 *
 * @author Pablo Alonso
 */
public class MultilayerPerceptron {
    
    private final LinkedList<double[]> neurons;
    private final LinkedList<double[]> weights;
    private final LinkedList<double[]> thresholds;
    private final LinkedList<double[]> summations;
    private final LinkedList<double[]> neuronsDelta;
    private final double alpha;
    
    /**
     *
     * @param inputSize the number of inputs this perceptron will have
     * @param outputSize the number of outputs this percentron will have
     * @param alpha the learning rate for the backpropagation phase
     * @param hiddenLayersSizes the number of neurons on each hidden layer
     */
    public MultilayerPerceptron(int inputSize, int outputSize, double alpha, int...hiddenLayersSizes){
        neurons = new LinkedList<>();
        weights = new LinkedList<>();
        thresholds = new LinkedList<>();
        summations = new LinkedList<>();
        neuronsDelta = new LinkedList<>();
        
        neurons.add(new double[inputSize]);
        for (int hiddenLayersSize : hiddenLayersSizes) {
            neurons.add(new double[hiddenLayersSize]);
        }
        neurons.add(new double[outputSize]);
        
        for (int i = 1; i < neurons.size(); i++) {
            weights.add(new double[neurons.get(i-1).length * neurons.get(i).length]);
            thresholds.add(new double[neurons.get(i).length]);
            summations.add(new double[neurons.get(i).length]);
            neuronsDelta.add(new double[neurons.get(i).length]);
        }
        
        this.alpha = alpha;
        
        initializeWeights();
        initializeThresholds(); 
    }
    
    /**
     *
     * @param inputSize the number of inputs this perceptron will have
     * @param outputSize the number of outputs this percentron will have
     * @param alpha the learning rate for the backpropagation phase
     * @param hiddenLayersSizes the number of neurons on each hidden layer
     * @param weights the weights of this perceptron. The array of double inside
     * the LinkedList will be modified if the programmer decide to train this perceptron
     * @param thresholds the thresholds of this perceptron. The array of double 
     * inside the LinkedList will be modified if the programmer decide to train this perceptron
     */
    public MultilayerPerceptron(int inputSize, int outputSize, double alpha, int[] hiddenLayersSizes, LinkedList<double[]> weights, LinkedList<double[]> thresholds){
        neurons = new LinkedList<>();
        this.weights = weights;
        this.thresholds = thresholds;
        summations = new LinkedList<>();
        neuronsDelta = new LinkedList<>();
        this.alpha = alpha;
        
        neurons.add(new double[inputSize]);
        for (int hiddenLayersSize : hiddenLayersSizes) {
            neurons.add(new double[hiddenLayersSize]);
        }
        neurons.add(new double[outputSize]);
        
        for (int i = 1; i < neurons.size(); i++) {
            summations.add(new double[neurons.get(i).length]);
            neuronsDelta.add(new double[neurons.get(i).length]);
        }
    }
    
    /**
     *
     * @param inputs The inputs used in the training phase, it must be the same 
     * size as the input size established in the creation of this Pereptron
     * @param expectedOutputs The outputs used in the training phase, it must be 
     * the same size as the output size established in the creation of this Pereptron
     */
    public void startTrainingCycle(double[] inputs, double[] expectedOutputs){
        setInputs(inputs);
        checkOutputSize(expectedOutputs);

        spreadInput();
        calculateOutputLayerNeuronsDelta(expectedOutputs);
        calculateHiddenLayersNeuronsDelta();
        updateWeigthsAndThresholds();
    }
    
    /**
     *
     * @return An Array with all the layers size. The first one will be the input
     * size, and the last one will be the output size.
     */
    public int[] getAllLayersSize(){
        int[] sizes = new int[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            sizes[i] = neurons.get(i).length;
        }
        return sizes;
    }

    public LinkedList<double[]> getWeights() {
        return weights;
    }

    public LinkedList<double[]> getThresholds() {
        return thresholds;
    }

    public double getAlpha() {
        return alpha;
    }
    
    /**
     *
     * @param inputs The inputs for this test, it must be the same size as the input size
     * established in the creation of this Pereptron
     * @param expectedOutputs The outputs for this test, it must be the same size as the output size
     * established in the creation of this Pereptron
     * @param errorRange The range of accepted output.
     * @return It will return true if all the outputs are in the range [(expectedOutput - errorRange), (expectedOutput + errorRange)]
     */
    public boolean testNeuralNet(double[] inputs, double[] expectedOutputs, double errorRange){
        setInputs(inputs);
        checkOutputSize(expectedOutputs);
        spreadInput();
        for (int i = 0; i < neurons.get(neurons.size() - 1).length; i++) {
            if(neurons.get(neurons.size() - 1)[i] < (expectedOutputs[i] - errorRange) || neurons.get(neurons.size() - 1)[i] > (expectedOutputs[i] + errorRange)) return false;
        }
        
        return true;
    }
    
    /**
     *
     * @param inputs The inputs for this test, it must be the same size as the input size
     * established in the creation of this Pereptron
     * @return The outputs obtained after spread the inputs
     */
    public double[] testNeuralNet(double[] inputs){
        setInputs(inputs);
        spreadInput();
        return neurons.get(neurons.size()-1);
    }
    
    private void setInputs(double[] inputs){
        if(inputs.length != neurons.get(0).length){
            throw new RuntimeException("The inputs passed as parameter is different in "
                    + "size from the inputs size set in the creation of this perceptron");
        }
        System.arraycopy(inputs, 0, neurons.get(0), 0, inputs.length);
    }
    
    private void checkOutputSize(double[] expectedOutputs){
        if(expectedOutputs.length != neurons.get(neurons.size() - 1).length){
            throw new RuntimeException("The output passed as parameter is different in "
                    + "size from the output size set in the creation of this perceptron");
        }
    }

    private void initializeWeights() {
        //Initialize all the weights with random values between -1 and 1
        Random random = new Random();
        for (double[] weight : weights) {
            for (int i = 0; i < weight.length; i++) {
                weight[i] = random.nextDouble() * 2 -1;
            }
        }
    }

    private void initializeThresholds() {
        //Initialize all the thresholds to 1
        for (double[] threshold : thresholds) {
            for (int i = 0; i < threshold.length; i++) {
                threshold[i] = 1;
            }
        }
    }
    
    private void spreadInput(){
        double summation = 0;
        double[] weigthList;
        double[] previousNeuronsList;
        int currentNeuronsListLength;
        double[] thresholdsList;
        double[] summationsList;
        double[] currentNeuronsList;
        for (int i = 1; i < neurons.size(); i++) {
            weigthList = weights.get(i-1);
            previousNeuronsList = neurons.get(i-1);
            currentNeuronsListLength = neurons.get(i).length;
            thresholdsList = thresholds.get(i-1);
            summationsList = summations.get(i-1);
            currentNeuronsList = neurons.get(i);
            for (int j = 0; j < currentNeuronsListLength; j++) {
                summation = 0;
                for (int k = 0; k < previousNeuronsList.length; k++) {
                    summation += weigthList[j+currentNeuronsListLength*k]*previousNeuronsList[k];
                }
                summation += thresholdsList[j];
                summationsList[j] = summation;
                currentNeuronsList[j] = Sigmoid.calculateSigmoid(summation);
            }
        }
    }
    
    private void calculateOutputLayerNeuronsDelta(double[] expectedOutputs){
        double[] deltaLayerList = neuronsDelta.get(neuronsDelta.size() - 1);
        double[] outputLayerList = neurons.get(neurons.size() - 1);
        double[] summationList = summations.get(summations.size() - 1);
        
        for (int i = 0; i < deltaLayerList.length; i++) {
            deltaLayerList[i] = -(expectedOutputs[i] - outputLayerList[i]) * Sigmoid.calculateDerivative(summationList[i]);
        }
    }
    
    private void calculateHiddenLayersNeuronsDelta(){
        double previousDeltaWeighted = 0;
        int weigthPointer = 0;
        int weigthLayerPointer = weights.size()-1;
        int summationsLayerPointer = summations.size() - 2;
        double[] currentDeltaList;
        double[] nextDeltaList;
        double[] weigthList;
        double[] summationsList;
        
        for (int i = neuronsDelta.size() - 2; i >= 0; i--) {
            weigthPointer = 0;
            currentDeltaList = neuronsDelta.get(i);
            nextDeltaList = neuronsDelta.get(i+1);
            weigthList = weights.get(weigthLayerPointer);
            summationsList = summations.get(summationsLayerPointer);
            for (int j = 0; j < currentDeltaList.length; j++) {
                previousDeltaWeighted = 0;
                for (int k = 0; k < nextDeltaList.length; k++) {
                    previousDeltaWeighted += nextDeltaList[k] * weigthList[weigthPointer];
                    weigthPointer++;
                }
                currentDeltaList[j] = Sigmoid.calculateDerivative(summationsList[j]) * previousDeltaWeighted;
            }
            weigthLayerPointer--;
            summationsLayerPointer--;
        }
    }
    
    private void updateWeigthsAndThresholds(){
        //Weigths
        int deltaLayerPointer = neuronsDelta.size() - 1;
        int neuronsLayerPointer = neurons.size() - 1;
        double[] weightsList;
        double[] neuronsDeltaList;
        int currentNeuronsLayerLength;
        double[] previousNeuronsLayerList;
        for (int i = weights.size() - 1; i >= 0; i--) {
            weightsList = weights.get(i);
            neuronsDeltaList = neuronsDelta.get(deltaLayerPointer);
            currentNeuronsLayerLength = neurons.get(neuronsLayerPointer).length;
            previousNeuronsLayerList = neurons.get(neuronsLayerPointer - 1);
            for (int j = 0; j < weightsList.length; j++) {
                weightsList[j] = (weightsList[j] - alpha * neuronsDeltaList[j % currentNeuronsLayerLength] * previousNeuronsLayerList[j / currentNeuronsLayerLength]);
            }
            deltaLayerPointer--;
            neuronsLayerPointer--;
        }
        
        //Thresholds
        deltaLayerPointer = neuronsDelta.size() - 1;
        double[] threshodlsList;
        for (int i = thresholds.size() - 1; i >= 0; i--) {
            neuronsDeltaList = neuronsDelta.get(deltaLayerPointer);
            threshodlsList = thresholds.get(i);
            for (int j = 0; j < thresholds.get(i).length; j++) {
                thresholds.get(i)[j] = threshodlsList[j] - alpha * neuronsDeltaList[j];
            }
            deltaLayerPointer--;
        }
    }
    
}
