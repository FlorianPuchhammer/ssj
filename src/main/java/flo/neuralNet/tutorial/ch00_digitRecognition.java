package flo.neuralNet.tutorial;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator.Set;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;



public class ch00_digitRecognition {
	
	public static void main(String[] args) throws IOException {
		int batchSize = 16;
		Set emnistSet = EmnistDataSetIterator.Set.BALANCED;
		EmnistDataSetIterator emnistTrain = new EmnistDataSetIterator(emnistSet,batchSize,true);
		EmnistDataSetIterator emnistTest = new EmnistDataSetIterator(emnistSet,batchSize,false);
		
		int outputNum = EmnistDataSetIterator.numLabels(emnistSet);
		int rngSeed = 123;
		int numRows = 28;
		int numCols = 28;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(rngSeed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam())
				.l2(1E-4)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(numRows * numCols)
						.nOut(1000)
						.activation(Activation.RELU)
						.weightInit(WeightInit.XAVIER)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nIn(1000)
						.nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false).backprop(true)
				.build();
		
		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		network.init();
		
		int eachIterations = 5;
		network.addListeners(new ScoreIterationListener(eachIterations));
						
//		 network.fit(emnistTrain);	
		 
//		 int numEpochs = 2;
//		 network.fit(new MultipleEpochsIterator(numEpochs, emnistTrain));
		
		Evaluation eval = network.evaluate(emnistTest);
		eval.accuracy();
		eval.precision();
		eval.recall();
		
		ROCMultiClass roc = network.evaluateROCMultiClass(emnistTest);
		roc.calculateAverageAUC();
		
		int classIndex = 0;
		roc.calculateAUC(classIndex);
		
		
		System.out.println(eval.stats());
		System.out.println(roc.stats());
		
		System.out.println("A   --  O K ! ! !");
		
	}

}
