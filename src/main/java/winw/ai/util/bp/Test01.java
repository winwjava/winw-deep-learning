package winw.ai.util.bp;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Test01 {

	public static void main(String[] args) throws IOException {
		var batchSize = 16; // how many examples to simultaneously train in the network
		var emnistSet = EmnistDataSetIterator.Set.BALANCED;
		//var emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true);
		var emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false);

		var outputNum = EmnistDataSetIterator.numLabels(emnistSet);// total output classes
		var rngSeed = 123; // integer for reproducability of a random number generator
		var numRows = 28; // number of "pixel rows" in an mnist digit
		var numColumns = 28;
		var conf = new NeuralNetConfiguration.Builder().seed(rngSeed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Adam()).l2(1e-4).list()
				.layer(new DenseLayer.Builder().nIn(numRows * numColumns) // Number of input datapoints.
						.nOut(1000) // Number of output datapoints.
						.activation(Activation.RELU) // Activation function.
						.weightInit(WeightInit.XAVIER) // Weight initialization.
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(1000)
						.nOut(outputNum).activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).build())
				// .pretrain(false).backprop(true)
				.build();

		// create the MLN
		var network = new MultiLayerNetwork(conf);
		network.init();

		// pass a training listener that reports score every 10 iterations
		var eachIterations = 5;
		network.addListeners(new ScoreIterationListener(eachIterations));

		// fit a dataset for a single epoch
		// network.fit(emnistTrain)

		// fit for multiple epochs
		// val numEpochs = 2
		// network.fit(new MultipleEpochsIterator(numEpochs, emnistTrain))

		// evaluate basic performance
		var eval = network.evaluate(emnistTest);
		eval.accuracy();
		eval.precision();
		eval.recall();

		// evaluate ROC and calculate the Area Under Curve
		var roc = network.evaluateROCMultiClass(emnistTest);
		roc.calculateAverageAUC();

		var classIndex = 0;
		roc.calculateAUC(classIndex);

		// optionally, you can print all stats from the evaluations
		System.out.println(eval.stats());
		System.out.println(roc.stats());
	}
}
