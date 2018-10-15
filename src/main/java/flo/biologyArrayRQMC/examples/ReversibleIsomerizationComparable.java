package flo.biologyArrayRQMC.examples;

import umontreal.ssj.rng.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import flo.neuralNet.NeuralNet;
import umontreal.ssj.markovchainrqmc.*;
import umontreal.ssj.util.PrintfFormat;
import umontreal.ssj.util.sort.MultiDim;
import umontreal.ssj.util.sort.MultiDim01;
import umontreal.ssj.probdist.DiscreteDistribution;
import umontreal.ssj.probdist.DiscreteDistributionInt;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.probdist.PoissonDist;
import umontreal.ssj.stat.TallyStore;

//class ReversibleisomerisationComparable extends MarkovChainComparable   implements MultiDim01{
public class ReversibleIsomerizationComparable extends ChemicalReactionNetwork {

	public ReversibleIsomerizationComparable(double[] c, double[] X0, double tau, double T) {
		this.c = c;
		this.X0 = X0;
		this.tau = tau;
		this.T = T;
		S = new double[][] { { -1, 1 }, { 1, -1 } };
		init();
	}

	public double getPerformance() {
		return X[1];

	}

	public double[] getState() {
		return X;
	}

	public int compareTo(MarkovChainComparable m, int i) {
		if (!(m instanceof ReversibleIsomerizationComparable)) {
			throw new IllegalArgumentException(
					"Can't compare an ReversibleIsomerization with other types of Markov chains.");
		}
		double mx;

		mx = ((ReversibleIsomerizationComparable) m).X[i];
		return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));

	}

	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append(" ReversibleIsomerisation:\n");
		sb.append("X0 =\t" + "{" + X0[0] + ", " + X0[1] + "}\n");
		sb.append("c =\t" + "{" + c[0] + ", " + c[1] + "}\n");
		sb.append("T =\t" + T + "\n");
		sb.append("tau =\t" + tau + "\n");
		sb.append("steps =\t" + numSteps + "\n");
		sb.append("----------------------------------------------\n\n");

		return sb.toString();
	}

	@Override
	public void computePropensities() {
		a[0] = c[0] * X[0];
		a[1] = c[1] * X[1];
	}
	
	private  MultiLayerNetwork genNetwork(int step, double lRate) {
		MultiLayerConfiguration conf = null;
		int seed = 123;
		WeightInit weightInit = WeightInit.NORMAL;
		Activation activation = Activation.RELU;
		Activation activation2 = Activation.RELU;
		LossFunction lossFunction = LossFunction.MSE;
		// AdaMax updater = new AdaMax(lRate);
		// AdaGrad updater = new AdaGrad(lRate);
		// Nesterovs updater = new Nesterovs(lRate);
		// RmsProp updater = new RmsProp(lRate);
		// updater.setLearningRateSchedule(new ExponentialSchedule(ScheduleType.EPOCH,
		// lRate, 0.9 ));

		IUpdater updater = new AdaDelta();

		int stateDim = this.getStateDimension();

		OptimizationAlgorithm optAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;

		
		
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0,new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation).build())
					.layer(1, new OutputLayer.Builder().nIn(stateDim).nOut(1)
							.activation(activation2).lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			
			return new MultiLayerNetwork(conf);
	}
	
	public static final void main(String[] args) throws IOException, InterruptedException {
		double epsInv = 1E2;
		double alpha = 1E-4;
		double[]c = {1.0,alpha};
		double[] x0 = {epsInv,epsInv/alpha};
		double T = 1.6;
		double tau = 0.2;
		
		ReversibleIsomerizationComparable model = new ReversibleIsomerizationComparable(c,x0,tau,T);
		model.init();
		System.out.println(model.toString());
		String filepath = "data/ReversibleIsometrization/";
		String dataLabel = "MCData";
		int numChains = 1048576/1;
		RandomStream stream = new MRG32k3a();

		
		
		NeuralNet net = new NeuralNet(model,filepath);
		
		net.genData(dataLabel, numChains, model.numSteps, stream);
	
//		net.genData(dataLabel, numChains, model.numSteps, stream);
		
		int batchSize = 128;
		int numEpochs = 32;

		/*
		 * READ DATA
		 */

		ArrayList<DataSet> dataAllList = new ArrayList<DataSet>();
		for(int s = 0; s < model.numSteps; s++) {
//			dataAllList.add(net.getData(dataLabel,s,numChains));
			dataAllList.add(net.getData(dataLabel,3,numChains));

		}
		
		
		/*
		 * GENERATE NETWORKS
		 */
		double lRate = 0.1;
		ArrayList<MultiLayerNetwork> networkList = new ArrayList<MultiLayerNetwork>();
		for (int i = 0; i < model.numSteps; i++) {
//			lRate += 1.0;
			networkList.add(net.genNetwork(model.numSteps-2,model.numSteps, lRate));
		}
		
		/*
		 * TRAIN NETWORK
		 */
		FileWriter fw = new FileWriter("./data/comparison.txt");
		StringBuffer sb = new StringBuffer("");
		String str;

		DataSet dataAll, trainingData, testData;
		double ratioTrainingData = 0.8;
		DataNormalization normalizer;
		MultiLayerNetwork network;
		SplitTestAndTrain testAndTrain;
		
		for(int i = 0; i < networkList.size(); i ++) {
			
			// GET DATA SET, SPLIT DATA, NORMALIZE
			dataAll = dataAllList.get(i);
			
			 testAndTrain = dataAll.splitTestAndTrain(ratioTrainingData);

			 trainingData = testAndTrain.getTrain();
			 testData = testAndTrain.getTest();
			
			normalizer = new NormalizerStandardize();
			normalizer.fit(trainingData);
			normalizer.transform(trainingData);
			normalizer.transform(testData);
			
			// GET AND TRAIN THE NETWORK
			network = networkList.get(i);
			
			str = "*******************************************\n";
			str += " CONFIGURATION: \n" + network.conf().toString() + "\n";
			str += "*******************************************\n";
			sb.append(str);
			System.out.println(str);

			NeuralNet.trainNetwork(network, trainingData, numEpochs, batchSize, (numChains / batchSize) * 1, 1000);
			
			// TEST NETWORK
			str = NeuralNet.testNetwork(network, testData, batchSize);
			sb.append(str);
			System.out.println(str);

		}
		
		fw.write(sb.toString());
		fw.flush();
		fw.close();


		
		
		
	}

}
