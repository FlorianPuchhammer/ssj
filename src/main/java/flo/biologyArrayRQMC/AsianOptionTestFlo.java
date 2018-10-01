package flo.biologyArrayRQMC;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.data.xml.DatasetReader;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import umontreal.ssj.markovchainrqmc.ArrayOfComparableChains;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.util.Chrono;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.MultiDimSort01;
import umontreal.ssj.util.sort.MultiDimSortN;

public class AsianOptionTestFlo extends ArrayOfComparableChains<AsianOptionComparable2> {

	public static String filepath = "./data/arrayRQMC_ML/asian/";
//	public static String filepath = "/u/puchhamf/misc/workspace/ssj/data/arrayRQMC_ML/asian/";
	public static MultiLayerNetwork network;

	public AsianOptionTestFlo(AsianOptionComparable2 baseChain) {
		super(baseChain);
	}

	public void genData(int n, int numSteps, RandomStream stream) throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		baseChain.simulRuns(n, numSteps, stream, states, performance);
		String fileName;
		StringBuffer sb;
		FileWriter file;
		for (int step = 0; step < numSteps; step++) {
			fileName = "MCData_Step_" + step + ".csv";
			sb = new StringBuffer("");
			file = new FileWriter(filepath + fileName);

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
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).weightInit(WeightInit.XAVIER)
				.updater(new Adam(lRate)).list()
				.layer(0,
						new DenseLayer.Builder().nIn(baseChain.getStateDimension()).nOut(1).activation(Activation.CUBE)
								.build())
				.layer(1, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
						.lossFunction(LossFunction.MSE).build())
				.pretrain(false).backprop(true).build();
		return new MultiLayerNetwork(conf);
	}

	public ArrayList<MultiLayerNetwork> genNetworkList(double lRate) {
		ArrayList<MultiLayerNetwork> netList = new ArrayList<MultiLayerNetwork>();
		Activation[] act = Activation.values();
		MultiLayerConfiguration conf;
		for (Activation a : act) {
			conf = new NeuralNetConfiguration.Builder().seed(123)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).weightInit(WeightInit.XAVIER)
					.updater(new AdaGrad(lRate)).list()
					.layer(0, new DenseLayer.Builder().nIn(baseChain.getStateDimension()).nOut(1).activation(a).build())
					.layer(1, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
							.lossFunction(LossFunction.MSE).build())
					.pretrain(false).backprop(true).build();
			netList.add(new MultiLayerNetwork(conf));
		}
		return netList;
	}

	public ArrayList<MultiLayerNetwork> genNetworkList() {
		int index = 0;
		double lRate = 0.5;
		int lRateVals = 10;
		Activation[] activations = Activation.values();
		WeightInit[] weightInits = { WeightInit.ZERO, WeightInit.ONES, WeightInit.SIGMOID_UNIFORM, WeightInit.NORMAL,
				WeightInit.LECUN_NORMAL, WeightInit.UNIFORM, WeightInit.XAVIER, WeightInit.XAVIER_UNIFORM,
				WeightInit.XAVIER_FAN_IN, WeightInit.XAVIER_LEGACY, WeightInit.RELU, WeightInit.RELU_UNIFORM,
//				WeightInit.IDENTITY, 
				WeightInit.LECUN_UNIFORM, WeightInit.VAR_SCALING_NORMAL_FAN_IN,
				WeightInit.VAR_SCALING_NORMAL_FAN_OUT, WeightInit.VAR_SCALING_NORMAL_FAN_AVG,
				WeightInit.VAR_SCALING_UNIFORM_FAN_IN, WeightInit.VAR_SCALING_UNIFORM_FAN_OUT,
				WeightInit.VAR_SCALING_UNIFORM_FAN_AVG };
		LossFunction[] lossFuncts = { LossFunction.MSE, LossFunction.L1, LossFunction.XENT, LossFunction.MCXENT,
				LossFunction.SQUARED_LOSS, LossFunction.RECONSTRUCTION_CROSSENTROPY, LossFunction.NEGATIVELOGLIKELIHOOD,
				LossFunction.COSINE_PROXIMITY, LossFunction.HINGE, LossFunction.SQUARED_HINGE,
				LossFunction.KL_DIVERGENCE, LossFunction.MEAN_ABSOLUTE_ERROR, LossFunction.L2,
				LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR, LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR,
				LossFunction.POISSON };
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
								.layer(0,
										new DenseLayer.Builder().nIn(baseChain.getStateDimension()).nOut(1)
												.activation(a).build())
								.layer(1, new OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
										.lossFunction(loss).build())
								.pretrain(false).backprop(true).build();
						netList.add(new MultiLayerNetwork(conf));
						if (index % 1000 == 0)
							System.out.println("Built index " + index);
						index++;
					}
		}

		return netList;

	}

	public static void main(String[] args) throws IOException, InterruptedException {
		double r = Math.log(1.09);
		int d = 4; // numSteps
		// double t1 = (240.0 - d + 1) / 365.0;
		// double T = 240.0 / 365.0;
		double t1 = 1.0 / d;
		double T = 1.0;
		double K = 100.0;
		double s0 = 100.0;
		double sigma = 0.5;

		int numChains = 65536;
		boolean genData = false;

		int batchSize = 128;
		int numEpochs = 32;

		int currentStep = d - 1;

		Chrono timer = new Chrono();
		RandomStream stream = new MRG32k3a();

		AsianOptionComparable2 asian = new AsianOptionComparable2(r, d, t1, T, K, s0, sigma);
		AsianOptionTestFlo test = new AsianOptionTestFlo(asian); // This is the array of comparable chains.

		if (genData) {
			timer.init();
			test.genData(numChains, d, stream);
			System.out.println("\n\nTiming:\t" + timer.format());
		}

		int linesToSkip = 0;
		char delimiter = ',';

		CSVRecordReader rr = new CSVRecordReader(linesToSkip, delimiter);
		rr.initialize(new FileSplit(new File(filepath + "MCData_Step_3.csv")));

		DataSetIterator iterAll = new RecordReaderDataSetIterator.Builder(rr, numChains).regression(2).build();
//		DataSetIterator iterAll = new RecordReaderDataSetIterator( rr,  batchSize, 2, 2,
//                true);
		DataNormalization normalizer = new NormalizerStandardize();

		DataSet dataAll = iterAll.next();
		normalizer.fit(dataAll);
		normalizer.transform(dataAll);

		SplitTestAndTrain testAndTrain = dataAll.splitTestAndTrain(0.8);

		DataSet trainingData = testAndTrain.getTrain();
		DataSet testData = testAndTrain.getTest();

		int maxItsTrain = trainingData.numExamples() / batchSize;
		int maxItsTest = testData.numExamples() / batchSize;

		double lRate = 5E-3;
		ArrayList<MultiLayerNetwork> networkList = test.genNetworkList();
		int index = 0;
//		Activation[] activations = Activation.values();

		FileWriter fw = new FileWriter("./data/comparison.txt");
		StringBuffer sb = new StringBuffer("");
		String str;
		ArrayList<Double> mseList = new ArrayList<Double>();

		Iterator<MultiLayerNetwork> networkIt = networkList.iterator();
		while (networkIt.hasNext()) {

//		network = test.genNetwork(d - 1, lRate);

			ScoreIterationListener listener = new ScoreIterationListener(maxItsTrain);
			network = networkIt.next();
			network.init();
			network.setListeners(listener);

			str = "*******************************************\n";
			str += " CONFIGURATION: \n" + network.conf().toString() + "\n";
			str += "*******************************************\n";
			sb.append(str);
			System.out.println(str);

			int its;
			for (int e = 0; e < numEpochs; e++) {

				trainingData.shuffle();

				List<DataSet> listDataTrain = trainingData.batchBy(batchSize);

				its = 0;
				while (listDataTrain.iterator().hasNext() && its++ < maxItsTrain)
					network.fit(listDataTrain.iterator().next());
//			System.out.println(listDataTrain.iterator().next().toString());
			}

			List<DataSet> listDataTest = testData.batchBy(batchSize);
			RegressionEvaluation eval = new RegressionEvaluation(1);
			its = 0;
			while (its++ < maxItsTest && listDataTest.iterator().hasNext()) {
				DataSet current = listDataTest.iterator().next();
				INDArray output = network.output(current.getFeatures());
				eval.eval(current.getLabels(), output);
			}
			mseList.add(eval.meanSquaredError(0));
			str = eval.stats() + "\n\n";
			sb.append(str);
			System.out.println(str);

			network.clear();
//			network.conf().clearVariables();
		}
		fw.write(sb.toString());
		fw.close();
		fw = new FileWriter("./data/comparison_mse.txt");
		fw.write(mseList.toString());
		fw.close();
	}
	

}
