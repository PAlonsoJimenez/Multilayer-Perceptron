package handwrittennumberidentifier;

/**
 *
 * @author Pablo Alonso
 */
public class HandwrittenNumberIdentifier {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        /*
        Loading a Perceptron data from a file. If you want to load a different 
        one, change the name to the one you want to load.
        */
        String name = "Perceptron Data Sample";
        NeuralNetWorker nnworker = new NeuralNetWorker(name);
        nnworker.testNeuralNet();
        
        /*
        To create a new Multilayer Perceptron uncomment the next lines and feel 
        free to change the alpha constant, the number of hidden layer and their 
        size, but have in mind that the class NeuralWorker use just one neuron 
        for the output. If you want to modify the output size, you should write 
        a new test and training method. 
        */
        //NeuralNetWorker nnworker = new NeuralNetWorker(0.2, 10, 20, 10);
        //nnworker.trainNeuralNet(100);
        /*
        If you want to save your perceptron, uncomment the next line and choose 
        a new name for your perceptron
        */
        //String name = ;
        //nnworker.saveNeuralNet(name);
    }
}
