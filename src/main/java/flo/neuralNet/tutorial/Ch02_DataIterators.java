package flo.neuralNet.tutorial;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;

public class Ch02_DataIterators {

	public static void main(String[] args) throws IOException   {
		int numRows = 28;
		int numCols = 28;
		
		int numOutput = 10;
		int batchSize = 128;
		int seed = 123;
		int numEpochs = 15;
		double lRate = 0.006;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Nesterovs(lRate,0.9))
				.list()
				.layer(0,new DenseLayer.Builder()
						.nIn(numRows * numCols)
						.nOut(1000)
						.activation(Activation.RELU)
						.weightInit(WeightInit.XAVIER)
						.build())
				.layer(1, new OutputLayer.Builder()
						.nIn(1000)
						.nOut(numOutput)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false).backprop(true)
				.build();
		
		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		
		network.setListeners(new ScoreIterationListener(1));
		
		
		DataSetIterator mnistTrainIterator = new MnistDataSetIterator(batchSize,true,seed);
		DataSetIterator mnistTestIterator = new MnistDataSetIterator(batchSize,false,seed);
				
		for(int e = 0; e < numEpochs; e++)
			network.fit(mnistTrainIterator);
//		network.fit(new MultipleEpochsIterator(numEpochs, mnistTrainIterator));
		
		Evaluation eval = network.evaluate(mnistTestIterator);
		System.out.println(eval.stats());
	}

}
