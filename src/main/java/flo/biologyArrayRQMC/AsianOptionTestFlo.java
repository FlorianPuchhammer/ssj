package flo.biologyArrayRQMC;

import java.io.FileWriter;
import java.io.IOException;

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
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import umontreal.ssj.markovchainrqmc.ArrayOfComparableChains;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.util.Chrono;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.MultiDimSort01;
import umontreal.ssj.util.sort.MultiDimSortN;

public class AsianOptionTestFlo extends ArrayOfComparableChains<AsianOptionComparable2> {
	
	public static String filepath = "/u/puchhamf/misc/workspace/ssj/data/arrayRQMC_ML/asian/";
	public static MultiLayerNetwork network;
	
	public AsianOptionTestFlo (AsianOptionComparable2 baseChain) {
		super(baseChain);
	}
	
	
	
	public  void genData(int n, int numSteps, RandomStream stream) throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		baseChain.simulRuns(n, numSteps, stream, states, performance);
		String fileName; 
		StringBuffer sb;
		FileWriter file;
		for (int step = 0; step < numSteps; step++) {
			fileName = "MCData_Step_" + step + ".csv";
			sb =  new StringBuffer("");
			file = new FileWriter(filepath + fileName);
			
			for(int i = 0; i < n; i++) {
				sb.append(i + ",");
				for(int j = 0; j < baseChain.getStateDimension();j++) 
					sb.append(states[i][step][j] + ",");
				sb.append(performance[i] + "\n");
			}
			file.write(sb.toString());
			file.close();
			System.out.println("*******************************************");
			System.out.println(" STEP " + step);
			System.out.println("*******************************************");
			System.out.println(sb.toString());
		}
	}
	
	public  MultiLayerNetwork genNetwork(int step, int batchSize, double lRate) {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.weightInit(WeightInit.XAVIER)
				.updater(new AdaGrad(lRate))
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(baseChain.getStateDimension())
						.nOut(1)
						.activation(Activation.IDENTITY)
						.build()
						)
				.layer(1, new OutputLayer.Builder()
						.nIn(1)
						.nOut(1)
						.activation(Activation.IDENTITY)
						.lossFunction(LossFunction.MSE)
						.build())
				.pretrain(false).backprop(true).build();
		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		network.addListeners(new ScoreIterationListener(1));
		return network;
				
	}
	
	public static void main (String[] args) throws IOException {
		double r = Math.log(1.09);
		int d = 4; //numSteps
		// double t1 = (240.0 - d + 1) / 365.0;
		// double T = 240.0 / 365.0;
		double t1 = 1.0 / d;
		double T = 1.0;
		double K = 100.0;
		double s0 = 100.0;
		double sigma = 0.5;
		
		int numChains = 65536;
		boolean genData = false;
		
		int nTest = 16384;
		int nTrain = 3 * nTest;
		int batchSize = 3*512;
		int numEpochs = 32;
		
		int currentStep = d-1;
		
		Chrono timer = new Chrono();
		RandomStream stream = new MRG32k3a();

		
		AsianOptionComparable2 asian = new AsianOptionComparable2 (r, d, t1, T, K,
				s0, sigma);
		AsianOptionTestFlo test = new AsianOptionTestFlo (asian);    // This is the array of comparable chains.
		
		if(genData) {
		timer.init();
		test.genData(numChains, d, stream);
		System.out.println("\n\nTiming:\t" + timer.format());
		}
		
		//train network
		//TODO:iterator
		
		System.out.println(asian.toString());
	}

}
