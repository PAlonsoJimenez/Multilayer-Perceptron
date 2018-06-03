package handwrittennumberidentifier;

/**
 *
 * @author Pablo Alonso
 */
public class Sigmoid {
    
    public static double calculateSigmoid(double x){
        return (1/(1+ Math.pow(Math.E, -x)));
    }
    
    public static double calculateDerivative(double x){
        double sigmoide = calculateSigmoid(x);
        return sigmoide*(1-sigmoide);
    }
}
