package flo.neuralNet;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import flo.biologyArrayRQMC.AsianOptionComparable2;
import flo.biologyArrayRQMC.AsianOptionTestFlo;
import flo.biologyArrayRQMC.examples.ChemicalReactionNetwork;
import flo.biologyArrayRQMC.examples.PKA;
import flo.biologyArrayRQMC.examples.ReversibleIsomerizationComparable;
import flo.biologyArrayRQMC.examples.SchloeglSystem;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.markovchainrqmc.MarkovChain;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.density.DEHistogram;
import umontreal.ssj.util.Chrono;

/**
 * Class to handle neural networks via DL4J. More precisely, it uses a
 * MarkovChainComparable as a basis to generate training- and test data, it
 * provides the possibility to generate neural networks, train and test them, as
 * well as to save and load them.
 * 
 * @author puchhamf
 *
 */
public class NeuralNet {

	/**
	 * Default location, to which neural networks, data, and normalizers are stored.
	 */
	public String filepath = "";
	public MarkovChainComparable model;

	/**
	 * Constructor for a specific \a model using the default location for saving.
	 * 
	 * @param model
	 *            the underlying model
	 */
	public NeuralNet(MarkovChainComparable model) {
		this(model, "");
	}

	/**
	 * Constructor for specific \a model and \a filepath.
	 * 
	 * @param model
	 *            the underlying model
	 * @param filepath
	 *            path to where networks, data, and normalizers are saved.
	 */
	public NeuralNet(MarkovChainComparable model, String filepath) {
		this.filepath = filepath;
		this.model = model;

	}

	/**
	 * Generates data (for training and testing) with the continuous stream (e.g.
	 * MC) \a stream. The chains are simulated for \a numSteps steps. The resulting
	 * files are Saved in <tt>.csv</tt> format. The string \a dataLabel is used as
	 * part of the file names. For each step one file is generated. Each line
	 * corresponds to one chain and the first columns contain the current state
	 * (before simulation) and the final performance in the last column.
	 * 
	 * @param dataLabel
	 *            String used for naming the data files.
	 * @param n
	 *            the number of chains
	 * @param numSteps
	 *            the number of steps
	 * @param stream
	 *            the random stream used
	 * @throws IOException
	 */
	public void genData(String dataLabel, int n, int numSteps, RandomStream stream) throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		model.simulRuns(n, numSteps, stream, states, performance);
		StringBuffer sb;
		FileWriter fw;
		File file;
		for (int step = 0; step < numSteps; step++) {
			sb = new StringBuffer("");
			file = new File(filepath + dataLabel + "_Step_" + step + ".csv");
//			file.getParentFile().mkdirs();
			fw = new FileWriter(file);

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < model.getStateDimension(); j++)
					sb.append(states[i][step][j] + ",");
				sb.append(performance[i] + "\n");
			}
			fw.write(sb.toString());
			fw.close();
			System.out.println("*******************************************");
			System.out.println(" STEP " + step);
			System.out.println("*******************************************");
			System.out.println(sb.toString());
		}
	}
	
	public void genDataPoly(String dataLabel, int n, int numSteps, RandomStream stream) throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		model.simulRunsWithSubstreams(n, numSteps, stream, states, performance);
		StringBuffer sb;
		FileWriter fw;
		File file;
		for (int step = 0; step < numSteps; step++) {
			sb = new StringBuffer("{");
			file = new File(filepath + dataLabel + "_Step_" + step + "poly.txt");
			file.getParentFile().mkdirs();
			fw = new FileWriter(file);

			for (int i = 0; i < n; i++) {
				sb.append("{");
				for (int j = 0; j < model.getStateDimension(); j++)
					sb.append(states[i][step][j] + ",");
				
				sb.append(performance[i] + "},\n");
			}
			sb.deleteCharAt(sb.lastIndexOf(","));
			sb.append("}");
			fw.write(sb.toString());
			fw.close();
			System.out.println("*******************************************");
			System.out.println(" STEP " + step);
			System.out.println("*******************************************");
			System.out.println(sb.toString());
		}
	}

	public static void genData(MarkovChainComparable model, String filePath, String dataLabel, int n, int numSteps, RandomStream stream) throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		model.simulRunsWithSubstreams(n, numSteps, stream, states, performance);
		StringBuffer sb;
		FileWriter fw;
		File file;
		for (int step = 0; step < numSteps; step++) {
			sb = new StringBuffer("");
			file = new File(filePath + dataLabel + "_Step_" + step + ".csv");
			file.getParentFile().mkdirs();
			fw = new FileWriter(file);
//			fw = new FileWriter(filePath + dataLabel + "_Step_" + step + ".csv");
			
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < model.getStateDimension(); j++)
					sb.append(states[i][step][j] + ",");
				sb.append(performance[i] + "\n");
			}
			fw.write(sb.toString());
			fw.close();
			System.out.println("*******************************************");
			System.out.println(" STEP " + step);
			System.out.println("*******************************************");
			System.out.println(sb.toString());
		}
	}
	
	//initX.length batches; l chains each to get dist, r draws each to get data.
	/*public void genDataWithDist(double[][] initX, int l, int r, RandomStream stream) {
		int stepsToEnd = model.numSteps;
		int k = initX.length;
//		double[] X0 = new double[k];
		double[][][] states = new double[l][][];
		double[] performance = new double[l];
		double[] u = new double[r];
		double[] draws = new double[r];
		
		DEHistogram histo;
		int numBins = (int) Math.round(Math.sqrt(l));
		for(int s = 0; s < model.numSteps; s++) {
			for(int kk = 0; kk < k; k++) {
//				Arrays.fill(X0,initX[kk]);
				model.setInitialState(initX[kk]);
//				model.setNumSteps(stepsToEnd);
				model.init();
				model.initialState();
				model.simulRuns(l, stepsToEnd, stream, states, performance);
				
				//CAREFUL: performance does not match states after sort!!!!
				Arrays.sort(performance);
				histo = new DEHistogram(performance,performance[0],performance[l-1],numBins);
				stream.nextArrayOfDouble(u, 0, r);
				Arrays.sort(u);
				histo.inverseF(u,draws);
			}//end kk
		}//end s
	}
	*/
	public void genData(String dataLabel, int n, int numSteps, RandomStream stream, int numDraws) throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		model.simulRuns(n, numSteps, stream, states, performance);
//		model.simulRunsWithSubstreams(n, numSteps, stream, states, performance);

		StringBuffer sb;
		FileWriter file;
		
		//gen performance dist
		int numBins = (int) Math.floor(Math.sqrt((double)n));
		double h = 1.0/(double) numBins;
		double [] data = performance.clone();
		Arrays.sort(data);
		DEHistogram histo = new DEHistogram(data, data[0],data[n-1],numBins);
//		draw numDraws values from performance dist
//		double[] pdf = histo.evalDensity();
//		double[] cdf = pdf.clone();
//		for(int i = 1; i < numBins; i++) {
//			cdf[i] = cdf[i-1] + pdf[i];
//		}
		
		double[] u = new double[numDraws];
		double[] draws = new double[numDraws];
		stream.nextArrayOfDouble(u, 0, numDraws);
		Arrays.sort(u);
		double pos = histo.getA() + h*0.5;
		double cdf = histo.evalDensity(pos);
		double cdfTemp, posTemp;
		for(int l = 0; l < numDraws; l++) {
			while(cdf < u[l]) {
				posTemp = pos + h;
				 cdfTemp = histo.evalDensity(posTemp);
				if(cdfTemp <= u[l]) {//if next step still not too large
					pos = posTemp;
					cdf = cdfTemp;
				}
				else { //if next step would be too large
					draws[l] = cdf;
					break;
				}
					
					
			} //goto
		}//end for l
		
		for (int step = 0; step < numSteps; step++) {
			sb = new StringBuffer("");
			file = new FileWriter(filepath + dataLabel + "_Step_" + step + ".csv");

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < model.getStateDimension(); j++)
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
	/**
	 * Same as #genData, but for a RandomStream \a stream which relies on substreams
	 * (i.e., QMC- or RQMC point sets).
	 * 
	 * @param dataLabel
	 *            String used for naming the data files.
	 * @param n
	 *            the number of chains
	 * @param numSteps
	 *            the number of steps
	 * @param stream
	 *            the random stream used
	 * @throws IOException
	 */
	public void genDataWithSubstreams(String dataLabel, int n, int numSteps, RandomStream stream) throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		model.simulRunsWithSubstreams(n, numSteps, stream, states, performance);
		StringBuffer sb;
		FileWriter file;
		for (int step = 0; step < numSteps; step++) {
			sb = new StringBuffer("");
			file = new FileWriter(filepath + dataLabel + "_Step_" + step + ".csv");

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < model.getStateDimension(); j++)
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

	/**
	 * Generates a neural network for the step \a step of the chain with learning
	 * rate \a lRate.
	 * 
	 * @param step
	 *            step of the chain
	 * @param lRate
	 *            learning rate
	 * @return a neural network
	 */
	
	public MultiLayerNetwork genNetwork(int step, int numSteps, double lRate) {
		MultiLayerConfiguration conf = null;
		int seed = 123;
		WeightInit weightInit = WeightInit.NORMAL;
		Activation activation1 = Activation.IDENTITY;
		Activation activation2 = Activation.IDENTITY;
		LossFunction lossFunction = LossFunction.MSE;
//		 AdaMax updater = new AdaMax(lRate);
//		 AdaGrad updater = new AdaGrad(lRate);
		// Nesterovs updater = new Nesterovs(lRate);
		// RmsProp updater = new RmsProp(lRate);
//		 updater.setLearningRateSchedule(new ExponentialSchedule(ScheduleType.EPOCH,
//		 lRate, 0.9 ));

		IUpdater updater = new AdaDelta();

		int stateDim = model.getStateDimension();

		OptimizationAlgorithm optAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
//		OptimizationAlgorithm optAlgo = OptimizationAlgorithm.LBFGS;

		ListBuilder interim = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
				.updater(updater).list();
		int numLayers = numSteps - step;
		int layerIndex;
		for(layerIndex = 0; layerIndex < numLayers-1; layerIndex++)
			interim.layer(layerIndex, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build());
		interim.layer(layerIndex,new OutputLayer.Builder().nIn(stateDim).nOut(1).activation(activation2)
				.lossFunction(lossFunction).build());
		conf = interim.pretrain(false).backprop(true).build();
		return new MultiLayerNetwork(conf);
	}
	/*private MultiLayerNetwork genNetwork(int step, double lRate) {
		MultiLayerConfiguration conf = null;
		int seed = 123;
		WeightInit weightInit = WeightInit.NORMAL;
		Activation activation1 = Activation.RELU;
		Activation activation2 = Activation.RELU;
		LossFunction lossFunction = LossFunction.MSE;
		// AdaMax updater = new AdaMax(lRate);
		// AdaGrad updater = new AdaGrad(lRate);
		// Nesterovs updater = new Nesterovs(lRate);
		// RmsProp updater = new RmsProp(lRate);
		// updater.setLearningRateSchedule(new ExponentialSchedule(ScheduleType.EPOCH,
		// lRate, 0.9 ));

		IUpdater updater = new AdaDelta();

		int stateDim = model.getStateDimension();

		OptimizationAlgorithm optAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
//		OptimizationAlgorithm optAlgo = OptimizationAlgorithm.LBFGS;

		switch (step) {
		case 3:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list().layer(0, new OutputLayer.Builder().nIn(stateDim).nOut(1)
							.activation(activation2).lossFunction(lossFunction).build())
					.pretrain(false).backprop(true).build();
		
			break;

		case 2:
			conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
					.updater(updater).list()
					.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
					.layer(1, new OutputLayer.Builder().nIn(stateDim).nOut(1).activation(activation2)
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
		// default:
		// conf = new
		// NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
		// .updater(updater).list()
		// .layer(0, new
		// DenseLayer.Builder().nIn(stateDim).nOut(1).activation(activation2).build())
		// .layer(1, new
		// OutputLayer.Builder().nIn(1).nOut(1).activation(Activation.IDENTITY)
		// .lossFunction(lossFunction).build())
		// .pretrain(false).backprop(true).build();
		// break;
		}

		return new MultiLayerNetwork(conf);
	}*/

	/**
	 * Another way to generate a NN...
	 * 
	 * @param step
	 * @param lRate
	 * @return
	 */
	public MultiLayerNetwork genNetwork2(int step, int numSteps, double lRate) {
		MultiLayerConfiguration conf = null;
		int seed = 123;
		WeightInit weightInit = WeightInit.NORMAL;
		Activation activation1 = Activation.IDENTITY;
		Activation activation2 = Activation.IDENTITY;
		LossFunction lossFunction = LossFunction.MSE;
		// AdaMax updater = new AdaMax(lRate);
		// AdaGrad updater = new AdaGrad(lRate);
		// Nesterovs updater = new Nesterovs(lRate);
		// RmsProp updater = new RmsProp(lRate);
		// updater.setLearningRateSchedule(new ExponentialSchedule(ScheduleType.EPOCH,
		// lRate, 0.9 ));

		IUpdater updater = new AdaDelta();

		int stateDim = model.getStateDimension();

		OptimizationAlgorithm optAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
//		OptimizationAlgorithm optAlgo = OptimizationAlgorithm.LBFGS;

		conf = new NeuralNetConfiguration.Builder().seed(seed).optimizationAlgo(optAlgo).weightInit(weightInit)
				.updater(updater).list()
				.layer(0, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
				.layer(1, new DenseLayer.Builder().nIn(stateDim).nOut(stateDim).activation(activation1).build())
				.layer(2, new DenseLayer.Builder().nIn(stateDim).nOut(1).activation(activation2).build())
				.layer(3, new OutputLayer.Builder().nIn(1).nOut(1).activation(activation1)
						.lossFunction(lossFunction).build())
				.pretrain(false).backprop(true).build();
		return new MultiLayerNetwork(conf);
	}

	/**
	 * Generates a list of networks using the learning rate \a lRate by calling
	 * #genNetwork for each step of the chain from 0 to \a numSteps.
	 * 
	 * @param lRate
	 *            the learning rate
	 * @param numSteps
	 *            the total number of steps of the chain.
	 * @return list of NNs
	 */
//	public ArrayList<MultiLayerNetwork> genNetworkList(double lRate, int numSteps) {
//		ArrayList<MultiLayerNetwork> netList = new ArrayList<MultiLayerNetwork>();
//		for (int step = 0; step < numSteps; step++) {
//			netList.add(genNetwork(step, lRate));
//		}
//		return netList;
//	}

	/**
	 * Same as #genNetworkList but with an individualized learning rate for each
	 * step.
	 * 
	 * @param lRateList
	 *            list containing the individual learning rates
	 * @param numSteps
	 *            the total number of steps of the chain
	 * @return list of NNs
	 */
//	public ArrayList<MultiLayerNetwork> genNetworkList(ArrayList<Double> lRateList, int numSteps) {
//		ArrayList<MultiLayerNetwork> netList = new ArrayList<MultiLayerNetwork>();
//		for (int step = 0; step < numSteps; step++) {
//			netList.add(genNetwork(step, lRateList.get(step)));
//		}
//		return netList;
//	}

	/**
	 * Generates several NNs by varying certain parameters.
	 * 
	 * @return
	 */
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
								.layer(0, new DenseLayer.Builder().nIn(model.getStateDimension())
										.nOut(model.getStateDimension()).activation(Activation.IDENTITY).build())
								.layer(1,
										new DenseLayer.Builder().nIn(model.getStateDimension()).nOut(1).activation(a)
												.build())
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

	/**
	 * Reads data from the files generated with #genData and makes them readable by
	 * the NNs
	 * 
	 * @param dataLabel
	 *            identifier for the file. Should be the same as the one with which
	 *            #genData was called.
	 * @param step
	 *            the current step of the chain.
	 * @param numData
	 *            the number of data considered (i.e. the number of chains).
	 * @return a data set.
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public DataSet getData(String dataLabel, int step, int numData) throws IOException, InterruptedException {
		int linesToSkip = 0;
		char delimiter = ',';

		CSVRecordReader rr = new CSVRecordReader(linesToSkip, delimiter);
		rr.initialize(new FileSplit(new File(filepath + dataLabel + "_Step_" + step + ".csv")));

		DataSetIterator iterAll = new RecordReaderDataSetIterator.Builder(rr, numData).regression(model.getStateDimension()-1).build();
		return iterAll.next();
	}

	/**
	 * Trains the NN \a network using the taining data \a trainingData over \a
	 * numEpochs epochs. At the beginning of each epoch the data is shuffled and,
	 * subsequently, \a maxItsTrain batches of size \a batchSize are drawn and used
	 * for training. The score of the training progress is displayed every \a
	 * printIterations iterations.
	 * 
	 * @param network
	 *            the NN to be trained.
	 * @param trainingData
	 *            data used for training.
	 * @param numEpochs
	 *            number of epochs.
	 * @param batchSize
	 *            the size of each batch.
	 * @param maxItsTrain
	 *            the maximum batches used per epoch.
	 * @param printIterations
	 *            how often the score of the training progress is displayed.
	 */
	public static void trainNetwork(MultiLayerNetwork network, DataSet trainingData, int numEpochs, int batchSize,
			int maxItsTrain, int printIterations) {
		network.init();
		ScoreIterationListener listener = new ScoreIterationListener(printIterations);
		network.setListeners(listener);

		// int maxItsTrain = trainingData.numExamples() / batchSize;

		int its;
		for (int e = 0; e < numEpochs; e++) {

			trainingData.shuffle();
			List<DataSet> listDataTrain = trainingData.batchBy(batchSize); 
			its = 0;
			// maxItsTrain=listDataTrain.size()/4;
			while (listDataTrain.iterator().hasNext() && its++ < maxItsTrain)
				network.fit(listDataTrain.iterator().next());

			network.setEpochCount(e);
		}
	}

	/**
	 * Tests the NN \a network with the data \a testData using batches of size \a
	 * batchSize. Here, the entire data is partitioned into batches once and each of
	 * these batches is used for testing.
	 * 
	 * @param network
	 * @param testData
	 * @param batchSize
	 * @return
	 */
	public static String testNetwork(MultiLayerNetwork network, DataSet testData, int batchSize) {
		List<DataSet> listDataTest = testData.batchBy(batchSize); // TODO: inefficient!
		RegressionEvaluation eval = new RegressionEvaluation(1);
		int maxItsTest = testData.numExamples() / batchSize;
		int its = 0;
		while (its++ < maxItsTest && listDataTest.iterator().hasNext()) {
			DataSet current = listDataTest.iterator().next();
			INDArray output = network.output(current.getFeatures());
			eval.eval(current.getLabels(), output);
		}
		String str = eval.stats();
		// System.out.println(str);
		return str;
	}

	/**
	 * Saves the NN \a net and the corresponding normalizer \a norma to a file
	 * composed as #filepath + \a networkLabel + <tt>.zip</tt>.
	 * 
	 * @param net the NN to be saved.
	 * @param networkLabel label identifying the network
	 * @param norma the normalizer to be saved.
	 * @throws IOException
	 */
	public void saveNetwork(MultiLayerNetwork net, String networkLabel, DataNormalization norma) throws IOException {
		File locationToSave = new File(filepath + networkLabel + ".zip"); // Where to save the network. Note: the file
																			// is in .zip format - can be opened
																			// externally
		boolean saveUpdater = true; // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you
									// want to train your network more in the future
		ModelSerializer.writeModel(net, locationToSave, saveUpdater, norma);

	}

	/**
	 * Loads the network saved at #filepath + \a networkLabel + <tt>.zip</tt>.
	 * @param networkLabel label identifying the network.
	 * @return the loaded network.
	 * @throws IOException
	 */
	public MultiLayerNetwork loadNetwork(String networkLabel) throws IOException {
		return ModelSerializer.restoreMultiLayerNetwork(filepath + networkLabel + ".zip");
	}

	/**
	 * Loads the normalizer saved at #filepath + \a networkLabel + <tt>.zip</tt>.
	 * @param networkLabel label identifying the file where the normalizer is saved.
	 * @return the loaded normalizer
	 */
	public DataNormalization loadNormalizer(String networkLabel) {
		return ModelSerializer.restoreNormalizerFromFile(new File(filepath + networkLabel + ".zip"));
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		/*
		 ***********************************************************************
		 ************* INITIALIZE THE MODEL*************************************
		 ***********************************************************************
		 */

//		double r = Math.log(1.09);
//		int d = 4; // numSteps
//		double t1 = 1.0 / d;
//		double T = 1.0;
//		double K = 100.0;
//		double s0 = 100.0;
//		double sigma = 0.5;
//		AsianOptionComparable2 model = new AsianOptionComparable2(r, d, t1, T, K, s0, sigma);
//		String dataFolder = "data/asian/";
		
		ChemicalReactionNetwork model;


//		double epsInv = 1E2;
//		double alpha = 1E-4;
//		double[]c = {1.0,alpha};
//		double[] x0 = {epsInv,epsInv/alpha};
//		double T = 1.6;
//		double tau = 0.2;
//
//		
//		
//		 model = new ReversibleIsomerizationComparable(c,x0,tau,T);
//		String dataFolder = "data/ReversibleIsometrization/";
//		model.init();
		
		
//		double[]c = {3E-7, 1E-4, 1E-3,3.5};
//		double[] x0 = {250.0, 1E5, 2E5};
//		double T = 4;
//		double tau = 0.2;
//
//		
//		
//		 model = new SchloeglSystem(c,x0,tau,T);
//		String dataFolder = "data/SchloeglSystem/";
//		model.init();
		
		double[]c = {8.696E-5, 0.02, 1.154E-4,0.02,0.016,0.0017};//Nano: 1E-9
		double[] x0 = {33000.0,33030.0, 1100.0, 1100.0, 1100.0, 1100.0};
		double T = 0.00005;
		double tau = T/20.0;

		
		
		 model = new PKA(c,x0,tau,T);
		String dataFolder = "data/PKA/";
		model.init();
		
		NeuralNet test = new NeuralNet(model,dataFolder); // This is the array of comparable chains.
		
		System.out.println(model.toString());

		
		
		int numChains = 524288 *2;
		int logNumChains = 19 + 1;

		
		Chrono timer = new Chrono();
		RandomStream stream = new MRG32k3a(); 
		
		

		/*
		 ***********************************************************************
		 ************* BUILD DATA***********************************************
		 ***********************************************************************
		 */
		boolean genData = false;

//		String dataLabel = "SobData";
		String dataLabel = "MCData";

//		PointSet sobol = new SobolSequence(logNumChains, 31, model.numSteps * model.getK());
//		PointSetRandomization rand = new LMScrambleShift(stream);
//		RQMCPointSet p = new RQMCPointSet(sobol, rand);

		if (genData) {
			timer.init();
//			test.genData(dataLabel, numChains, model.numSteps, p.iterator());
			test.genData(dataLabel, numChains, model.numSteps, stream);
			System.out.println("\n\nTiming:\t" + timer.format());
		}
		/*
		 ***********************************************************************
		 ************* NEURAL NETWORK*******************************************
		 ***********************************************************************
		 */

		int batchSize = 128;
		int numEpochs = 32;

		/*
		 * READ DATA
		 */

		ArrayList<DataSet> dataAllList = new ArrayList<DataSet>();
		for(int s = 0; s < model.numSteps; s++) {
			dataAllList.add(test.getData(dataLabel,s,numChains));
		}
		
		
		/*
		 * GENERATE NETWORKS
		 */
		double lRate = 0.1;
		ArrayList<MultiLayerNetwork> networkList = new ArrayList<MultiLayerNetwork>();
		for (int i = 0; i < model.numSteps; i++) {
//			lRate += 1.0;
			networkList.add(test.genNetwork2(model.numSteps-2,model.numSteps, lRate));
		}
		
		/*
		 * TRAIN NETWORK
		 */
		FileWriter fw = new FileWriter("./data/comparison" +dataLabel+ ".txt");
		StringBuffer sb = new StringBuffer("");
		String str;

		DataSet dataAll, trainingData, testData;
		double ratioTrainingData = 0.8;
		DataNormalization normalizer;
		MultiLayerNetwork network;
		SplitTestAndTrain testAndTrain;
		
		for(int i = 0; i < model.numSteps; i ++) {
			
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
			str += " CONFIGURATION: \n" + network.getLayerWiseConfigurations().toString() + "\n";
			str += "*******************************************\n";
			sb.append(str);
			System.out.println(str);

			NeuralNet.trainNetwork(network, trainingData, numEpochs, batchSize, (numChains / batchSize) *2, 1000);
			
			// TEST NETWORK
			str = NeuralNet.testNetwork(network, testData, batchSize);
			sb.append(str);
			System.out.println(str);


			// saveNetwork(network,"Asian_Step" + currentStep, normalizer);
			test.saveNetwork(network, dataLabel + "Step" + i, normalizer);
			// i++;
			// network.clear();
			// network = loadNetwork("Asian_Step" + currentStep);
			//
			// System.out.println(network.getLayerWiseConfigurations().toString());
			// System.out.println(loadNormalizer("Asian_Step"+currentStep).toString());
		}

		fw.write(sb.toString());
		fw.flush();
		fw.close();

	}

}
