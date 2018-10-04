package flo.biologyArrayRQMC;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.datasets.iterator.MultiDataSetIteratorSplitter;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.solvers.StochasticGradientDescent;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.data.xml.DatasetReader;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.AdaMax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import umontreal.ssj.markovchainrqmc.ArrayOfComparableChains;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.util.Chrono;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.MultiDimSort01;
import umontreal.ssj.util.sort.MultiDimSortN;

public class AsianOptionTestFlo extends ArrayOfComparableChains<AsianOptionComparable2> {

	public static String filepath = "./data/asian/";
	// public static String filepath =
	// "/u/puchhamf/misc/workspace/ssj/data/arrayRQMC_ML/asian/";
	public static MultiLayerNetwork network;

	public AsianOptionTestFlo(AsianOptionComparable2 baseChain) {
		super(baseChain);
	}

	public void genData(String dataLabel, int n, int numSteps, RandomStream stream) throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		baseChain.simulRuns(n, numSteps, stream, states, performance);
		StringBuffer sb;
		FileWriter file;
		for (int step = 0; step < numSteps; step++) {
			sb = new StringBuffer("");
			file = new FileWriter(filepath + dataLabel + "_Step_" + step + ".csv");

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < baseChain.getStateDimension(); j++)
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

	public MultiLayerNetwork genNetwork(int step, double lRate) {
		MultiLayerConfiguration conf = null;
		int seed = 123;
		WeightInit weightInit = WeightInit.NORMAL;
		Activation activation1 = Activation.IDENTITY;
		Activation activation2 = Activation.RELU;
		LossFunction lossFunction = LossFunction.MSE;
//		 AdaGrad updater = new AdaGrad(lRate);
		IUpdater updater = new AdaDelta();

		int stateDim = baseChain.getStateDimension();

		OptimizationAlgorithm optAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;

		switch (step) {
		case 3:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new OutputLayer.Builder().nIn(stateDim).nOut(1).activation(activation2)
							.lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			break;

		case 2:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(1, new OutputLayer.Builder().nIn(1).nOut(stateDim).activation(activation2)
							.lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			break;
		case 1:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(1, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(2, new OutputLayer.Builder().nIn(stateDim).nOut(1).activation(activation2)
							.lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			break;
		case 0:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(1, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(2, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(3, new OutputLayer.Builder().nIn(stateDim).nOut(1).activation(activation2)
							.lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			break;
//		default:
//			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
//					.updater(updater).list()
//					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(1).activation(activation2).build())
//					.layer(1, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
//							.lossFunction(lossFunction).build())
//					.pretrain(false).backprop(true).build();
//			break;
		}

		return new MultiLayerNetwork(conf);
	}
	
	public MultiLayerNetwork genNetwork2(int step, double lRate) {
		MultiLayerConfiguration conf;
		int seed = 123;
		WeightInit weightInit = WeightInit.NORMAL;
		Activation activation1 = Activation.IDENTITY;
		Activation activation2 = Activation.HARDSIGMOID;
		LossFunction lossFunction = LossFunction.MSE;
//		 AdaGrad updater = new AdaGrad(lRate);
		IUpdater updater = new AdaDelta();

		int stateDim = baseChain.getStateDimension();

		OptimizationAlgorithm optAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;

		switch (step) {
		case 3:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(1).activation(activation2).build())
					.layer(1, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
							.lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			break;

		case 2:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(1, new DenseLayer.Builder().nIn(stateDim).nOut(1).activation(activation2).build())
					.layer(2, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
							.lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			break;
		case 1:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(1, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(2, new DenseLayer.Builder().nIn(stateDim).nOut(1).activation(activation2).build())
					.layer(3, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
							.lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			break;
		case 0:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(1, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(2, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(3, new DenseLayer.Builder().nIn(stateDim).nOut(1).activation(activation2).build())
					.layer(4, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
							.lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			break;
		default:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(1).activation(activation2).build())
					.layer(1, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
							.lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
			break;
		}

		return new MultiLayerNetwork(conf);
	}

	public  ArrayList<MultiLayerNetwork> genNetworkList(double lRate) {
		ArrayList<MultiLayerNetwork> netList = new ArrayList<MultiLayerNetwork>();
		for (int step = 0; step < baseChain.d; step++) {
			netList.add(genNetwork(step, lRate));
		}
		return netList;
	}

	public ArrayList<MultiLayerNetwork> genNetworkList(double[] lRateArray) {
		ArrayList<MultiLayerNetwork> netList = new ArrayList<MultiLayerNetwork>();
		for (int step = 0; step < baseChain.d; step++) {
			netList.add(genNetwork(step, lRateArray[step]));
		}
		return netList;
	}

	public ArrayList<MultiLayerNetwork> genNetworkListGridSearch() {
		int index = 0;
		double lRate = 0.5;
		int lRateVals = 5;
		Activation[] activations = {
				// Activation.CUBE,
				// Activation.ELU,
				// Activation.HARDSIGMOID,
				// Activation.HARDTANH,
				Activation.IDENTITY,
				// Activation.LEAKYRELU,
				// Activation.RATIONALTANH,
				Activation.RELU,
				// Activation.RELU6,
				Activation.RRELU, Activation.SIGMOID,
				// Activation.SOFTMAX,
				Activation.SOFTPLUS,
				// Activation.SOFTSIGN,
				Activation.TANH
				// Activation.RECTIFIEDTANH,
				// Activation.SELU,
				// Activation.SWISH,
				// Activation.THRESHOLDEDRELU
		};
		WeightInit[] weightInits = {
				// WeightInit.ZERO,
				// WeightInit.ONES,
				// WeightInit.SIGMOID_UNIFORM,
				// WeightInit.NORMAL,
				// WeightInit.LECUN_NORMAL,
				WeightInit.UNIFORM, WeightInit.XAVIER,
				// WeightInit.XAVIER_UNIFORM,
				// WeightInit.XAVIER_FAN_IN,
				// WeightInit.XAVIER_LEGACY,
				WeightInit.RELU
				// WeightInit.RELU_UNIFORM,
				// WeightInit.IDENTITY,
				// WeightInit.LECUN_UNIFORM,
				// WeightInit.VAR_SCALING_NORMAL_FAN_IN,
				// WeightInit.VAR_SCALING_NORMAL_FAN_OUT,
				// WeightInit.VAR_SCALING_NORMAL_FAN_AVG,
				// WeightInit.VAR_SCALING_UNIFORM_FAN_IN,
				// WeightInit.VAR_SCALING_UNIFORM_FAN_OUT,
				// WeightInit.VAR_SCALING_UNIFORM_FAN_AVG
		};
		LossFunction[] lossFuncts = { LossFunction.MSE,
				// LossFunction.L1,
				// LossFunction.XENT,
				// LossFunction.MCXENT,
				// LossFunction.SQUARED_LOSS,
				// LossFunction.RECONSTRUCTION_CROSSENTROPY,
				// LossFunction.NEGATIVELOGLIKELIHOOD,
				// LossFunction.COSINE_PROXIMITY,
				// LossFunction.HINGE,
				// LossFunction.SQUARED_HINGE,
				// LossFunction.KL_DIVERGENCE,
				// LossFunction.MEAN_ABSOLUTE_ERROR,
				// LossFunction.L2,
				// LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
				// LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR,
				// LossFunction.POISSON
		};
		System.out.println("TOTAL:\t" + activations.length * weightInits.length * lossFuncts.length * lRateVals);
		MultiLayerConfiguration conf;
		ArrayList<MultiLayerNetwork> netList = new ArrayList<MultiLayerNetwork>();
		for (int l = 0; l < lRateVals; l++) {
			lRate /= 2.0;
			for (Activation a : activations)
				for (WeightInit w : weightInits)
					for (LossFunction loss : lossFuncts) {
						conf = new NeuralNetConfiguration.Builder().seed(123)
								.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).weightInit(w)
								.updater(new AdaGrad(lRate)).list()
								.layer(0, new DenseLayer.Builder().nIn(baseChain.getStateDimension())
										.nOut(baseChain.getStateDimension()).activation(Activation.IDENTITY).build())
								.layer(1,
										new DenseLayer.Builder().nIn(baseChain.getStateDimension()).nOut(1)
												.activation(a).build())
								.layer(2, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
										.lossFunction(loss).build())
								.pretrain(false).backprop(true).build();
						netList.add(new MultiLayerNetwork(conf));

						if (index % 10 == 0)
							System.out.println("Built index " + index);
						index++;
					}
		}

		return netList;

	}
	
	public static DataSet getData(String dataLabel, int step, int numData) throws IOException, InterruptedException {
		int linesToSkip = 0;
		char delimiter = ',';

		CSVRecordReader rr = new CSVRecordReader(linesToSkip, delimiter);
		rr.initialize(new FileSplit(new File(filepath + dataLabel + "_Step_" + step + ".csv")));

		DataSetIterator iterAll = new RecordReaderDataSetIterator.Builder(rr, numData).regression(2).build();
		return iterAll.next();
	}
	
	public static void trainNetwork(MultiLayerNetwork network, DataSet trainingData, int numEpochs, int batchSize, int maxItsTrain,int printIterations) {
		network.init();
		ScoreIterationListener listener = new ScoreIterationListener(printIterations);
		network.setListeners(listener);
		
//		int maxItsTrain = trainingData.numExamples() / batchSize;
		
		int its;
		for (int e = 0; e < numEpochs; e++) {

			trainingData.shuffle();
			List<DataSet> listDataTrain = trainingData.batchBy(batchSize); //TODO: very inefficient!
			its = 0;
//			maxItsTrain=listDataTrain.size()/4;
			while (listDataTrain.iterator().hasNext() && its++ < maxItsTrain)
				network.fit(listDataTrain.iterator().next());
			
			network.setEpochCount(e);
		}
	}
	
	public static String testNetwork(MultiLayerNetwork network, DataSet testData, int batchSize) {
		List<DataSet> listDataTest = testData.batchBy(batchSize); //TODO: inefficient!
		RegressionEvaluation eval = new RegressionEvaluation(1);
		int maxItsTest = testData.numExamples() / batchSize;
		int its = 0;
		while (its++ < maxItsTest && listDataTest.iterator().hasNext()) {
			DataSet current = listDataTest.iterator().next();
			INDArray output = network.output(current.getFeatures());
			eval.eval(current.getLabels(), output);
		}
		String str = eval.stats();
//		System.out.println(str);
		return str;
	}
	
	public static void saveNetwork( MultiLayerNetwork net, String networkLabel,DataNormalization norma) throws IOException {
		  File locationToSave = new File(filepath + networkLabel + ".zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
	        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
	        ModelSerializer.writeModel(net, locationToSave, saveUpdater,norma);
	        
	}
	
	public static MultiLayerNetwork loadNetwork(String networkLabel) throws IOException {
		return ModelSerializer.restoreMultiLayerNetwork(filepath + networkLabel + ".zip");
	}
	
	public static DataNormalization loadNormalizer (String networkLabel) {
		return ModelSerializer.restoreNormalizerFromFile(new File(filepath + networkLabel + ".zip"));
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		/*
		 ***********************************************************************
		 ************* INITIALIZE THE MODEL*************************************
		 ***********************************************************************
		 */

		double r = Math.log(1.09);
		int d = 4; // numSteps
		// double t1 = (240.0 - d + 1) / 365.0;
		// double T = 240.0 / 365.0;
		double t1 = 1.0 / d;
		double T = 1.0;
		double K = 100.0;
		double s0 = 100.0;
		double sigma = 0.5;
		int numChains = 524288;

		Chrono timer = new Chrono();
		RandomStream stream = new MRG32k3a();

		AsianOptionComparable2 asian = new AsianOptionComparable2(r, d, t1, T, K, s0, sigma);

		System.out.println(asian.toString());

		AsianOptionTestFlo test = new AsianOptionTestFlo(asian); // This is the array of comparable chains.

		/*
		 ***********************************************************************
		 ************* BUILD DATA***********************************************
		 ***********************************************************************
		 */
		boolean genData = false;

		String dataLabel = "MCData";

		if (genData) {
			timer.init();
			test.genData(dataLabel, numChains, d, stream);
			System.out.println("\n\nTiming:\t" + timer.format());
		}
		/*
		 ***********************************************************************
		 ************* NEURAL NETWORK*******************************************
		 ***********************************************************************
		 */

		int currentStep = 1;

		int batchSize = 128;
		int numEpochs = 32;

		/*
		 * READ DATA
		 */


		DataSet dataAll = getData(dataLabel, currentStep,numChains);
		
		

		/*
		 * SPLIT DATA, DEFINE ITERATORS, AND NORMALIZE
		 */

		

		double ratioTrainingData = 0.8;
		SplitTestAndTrain testAndTrain = dataAll.splitTestAndTrain(ratioTrainingData);

		DataSet trainingData = testAndTrain.getTrain();
		DataSet testData = testAndTrain.getTest();

		DataNormalization normalizer = new NormalizerStandardize();
		// DataNormalization normalizer = new NormalizerMinMaxScaler();

		normalizer.fit(trainingData);
		normalizer.transform(trainingData);
		normalizer.transform(testData);

	

		
		double lRate = 64;
//		lRate = 8.0;
//		 ArrayList<MultiLayerNetwork> networkList = test.genNetworkList();
		ArrayList<MultiLayerNetwork> networkList = new ArrayList<MultiLayerNetwork>();
//		for (int i = 0; i < 7; i++) {
//			lRate *= 0.5;
//
			networkList.add(test.genNetwork(currentStep, lRate));
//		}
		FileWriter fw = new FileWriter("./data/comparison_id+act.txt");
		StringBuffer sb = new StringBuffer("");
		String str;



		Iterator<MultiLayerNetwork> networkIt = networkList.iterator();
		MultiLayerNetwork network;
		while (networkIt.hasNext()) {

			network = networkIt.next();
			
			str = "*******************************************\n";
			str += " CONFIGURATION: \n" + network.conf().toString() + "\n";
			str += "*******************************************\n";
			sb.append(str);
			System.out.println(str);
			
			
			
			trainNetwork(network,trainingData,numEpochs, batchSize, (numChains / batchSize) * 4,1000);
			
			str = testNetwork(network,testData,batchSize);
			sb.append(str);
			System.out.println(str);
			
					
			
//			saveNetwork(network,"Asian_Step" + currentStep, normalizer);
//			network.clear();
//			network = loadNetwork("Asian_Step" + currentStep);
//			
//			System.out.println(network.getLayerWiseConfigurations().toString());
//			System.out.println(loadNormalizer("Asian_Step"+currentStep).toString());
		}

		fw.write(sb.toString());
		fw.flush();
		fw.close();

	}

}
