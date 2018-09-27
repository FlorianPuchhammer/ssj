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
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Ch03_FeedForward {

	public static void main(String[] args) throws IOException {
		int seed = 12345;
		int numInput = 28*28;
		int numOutput = 10;
		int numTrans = 250;
		double lRate = 0.05;
		int numEpochs = 5;
		int batchSize = 128;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.activation(Activation.RELU)
				.updater(new AdaGrad(lRate))
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.l2(0.0001)
				.list()
				.layer(0,new DenseLayer.Builder().nIn(numInput).nOut(numTrans).weightInit(WeightInit.XAVIER).activation(Activation.SIGMOID)
						.build())
//				.layer(1,new DenseLayer.Builder().nIn(numTrans).nOut(numTrans*2).weightInit(WeightInit.XAVIER).activation(Activation.SIGMOID)
//						.build())
//				.layer(2,new DenseLayer.Builder().nIn(numTrans*2).nOut(numTrans).weightInit(WeightInit.XAVIER).activation(Activation.SIGMOID)
//						.build())
				.layer(1,new OutputLayer.Builder().nIn(numTrans).nOut(numOutput).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD).build())
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
				
		System.out.println("A  --  O K !");
	}
}
