package winw.ai.util.bp;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;

public class Test {
	
	public static void main(String[] args) {
	    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .weightInit(WeightInit.XAVIER)
	            .activation(Activation.RELU)
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .updater(new Sgd(0.05))
	            // ... other hyperparameters
	            .list()
	            //.backprop(true)
	            .build();
	    
	    conf.getCacheMode();
	}

}
