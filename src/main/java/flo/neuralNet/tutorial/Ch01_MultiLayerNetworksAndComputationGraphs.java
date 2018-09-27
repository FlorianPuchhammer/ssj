package flo.neuralNet.tutorial;


import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.*;



public class Ch01_MultiLayerNetworksAndComputationGraphs {

	public static void main(String[] args) {
		int seed = 123;
		double lRate = 0.1;
		int numIts = 3;
		
		MultiLayerConfiguration conf =  new NeuralNetConfiguration.Builder()
				.seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Nesterovs(lRate,0.9))
				.list()
				.layer(0,new DenseLayer.Builder().nIn(784).nOut(100).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build())
				.layer(1,new OutputLayer.Builder().nIn(100).nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.CUBE).build() )
				.pretrain(false).backprop(false)
				.build();
		
		System.out.println(conf.toString());
		
		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		
//		network.setLearningRate(1.0);
//		network.setLearningRate(0, 0.75);
		
		System.out.println(network.toString());
		
		System.out.println("A   --  O K ! ! !");
				
	}
}
