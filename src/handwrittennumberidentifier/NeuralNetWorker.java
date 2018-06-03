package handwrittennumberidentifier;

/**
 *
 * @author Pablo Alonso
 */
public class NeuralNetWorker {
    
    private final MultilayerPerceptron perceptron;
    
    /**
     *
     * @param alpha The Alpha constant of the new perceptron
     * @param hiddenLayersSizes The sizes of the hiddenLayer(s) of the new perceptron
     */
    public NeuralNetWorker(double alpha, int...hiddenLayersSizes){
        perceptron = new MultilayerPerceptron(784, 1, alpha, hiddenLayersSizes);
    }
    
    /**
     *
     * @param perceptronName Name of the Perceptron that will be loaded
     */
    public NeuralNetWorker(String perceptronName){
        perceptron = DataManager.loadPerceptron(perceptronName);
    }
    
    public void saveNeuralNet(String perceptronName){
        DataManager.savePercepron(perceptron, perceptronName, getHits(DataManager.loadTestData(), DataManager.loadTestLabels()), getHits(DataManager.loadTrainData(), DataManager.loadTrainLabels()));
    }
    
    public void trainNeuralNet(int numberOfCycles){
        double[][] trainDataSet = DataManager.loadTrainData();
        double[][] testDataSet = DataManager.loadTestData();
        double[] trainLabels = DataManager.loadTrainLabels();
        double[] testLabels = DataManager.loadTestLabels();
        
        for (int i = 0; i < numberOfCycles; i++) {
            for (int j = 0; j < trainDataSet.length; j++) {
                perceptron.startTrainingCycle(trainDataSet[j], new double[]{trainLabels[j]});
            }
            int trainingDataHits = getHits(trainDataSet, trainLabels);
            int testDataHits = getHits(testDataSet, testLabels);
            System.out.println("Cycle: " + i + ", Training Data Hits: " + trainingDataHits + ", Test Data Hits: " + testDataHits);
        }
    }
    
    public void testNeuralNet(){
        int trainingDataHits = getHits(DataManager.loadTrainData(), DataManager.loadTrainLabels());
        int testDataHits = getHits(DataManager.loadTestData(), DataManager.loadTestLabels());
        System.out.println("Training Data Hits: " + trainingDataHits + ", Test Data Hits: " + testDataHits);
    }
    
    private int getHits(double[][] inputMatrix, double[] output){
        int hits = 0;
        for (int i = 0; i < inputMatrix.length; i++) {
            if(perceptron.testNeuralNet(inputMatrix[i], new double[]{output[i]}, 0.03)) hits++;
        }
        return hits;
    }
}
