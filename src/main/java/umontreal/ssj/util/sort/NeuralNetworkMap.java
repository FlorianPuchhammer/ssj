/*
 * Class:        HilbertCurveMap
 * Description:  Map the Hilbert curve in a d-dimensional space [0,1)^d.
 * Environment:  Java
 * Software:     SSJ 
 * Copyright (C) 2014  Pierre L'Ecuyer and Universite de Montreal
 * Organization: DIRO, Universite de Montreal
 * @author       
 * @since

 * SSJ is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (GPL) as published by the
 * Free Software Foundation, either version 3 of the License, or
 * any later version.

 * SSJ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * A copy of the GNU General Public License is available at
   <a href="http://www.gnu.org/licenses">GPL licence site</a>.
 */

/* IMPORTANT NOTE:
* Much of this code has been taken (with adaptations) from  
*     the hilbert.c  code  
* Author: Spencer W. Thomas
* EECS Dept.
* University of Michigan
* Date: Thu Feb  7 1991
* Copyright (c) 1991, University of Michigan
*/
package umontreal.ssj.util.sort;

import java.util.Comparator;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

public class NeuralNetworkMap {

	/*
	 * public static final int seed = 12345; //Number of epochs (full passes of the
	 * data) public static final int nEpochs = 200; //Number of data points //public
	 * static final int nSamples = 1000; public static final int nSamples = 4000;
	 * //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
	 * public static final int batchSize = 100; //Network learning rate public
	 * static final double learningRate = 0.01;
	 * 
	 * 
	 * public static final Random rng = new Random(seed);
	 */
	public int seed;
	// Number of epochs (full passes of the data)
	public int nEpochs;
	int batchSize;

	// Batch size: i.e., each epoch has nSamples/batchSize parameter updates

	// Network learning rate
	public double learningRate;
	public int numInputs, numOutputs, numHiddenNodes;
	public static Random rng;
	public String fileTrain, fileTest;
	public static MultiLayerNetwork net;
	public static DataSetIterator iterTrain;
	public static DataSetIterator iterTest;
	public static EarlyStoppingResult result;

	public NeuralNetworkMap(String fileTrain, String fileTest, int numInputs, int numOutputs, int numHiddenNodes,
			int seed, double learningRate, int nEpochs, int batchSize) {
		this.numHiddenNodes = numHiddenNodes;
		this.numInputs = numInputs;
		this.numOutputs = numOutputs;
		this.seed = seed;
		this.nEpochs = nEpochs;
		this.learningRate = learningRate;
		rng = new Random(seed);
		this.fileTrain = fileTrain;
		this.fileTest = fileTest;
		this.batchSize = batchSize;
	}

	public static double[][] transposeMatrix(double[][] m) {
		double[][] tmp = new double[m[0].length][m.length];
		for (int i = 0; i < m[0].length; i++)
			for (int j = 0; j < m.length; j++)
				tmp[i][j] = m[j][i];
		return tmp;
	}
	/*
	 * public static DataSetIterator generateIterator( String filename, int
	 * batchSize) throws IOException, InterruptedException { SequenceRecordReader
	 * trainReader = new CSVSequenceRecordReader(0, ","); trainReader.initialize(new
	 * FileSplit( new File (filename))); DataSetIterator Iter = new
	 * SequenceRecordReaderDataSetIterator(trainReader, batchSize, -1, 1, true);
	 * RecordReader trainReader = new CSVRecordReader(); trainReader.initialize(new
	 * FileSplit( new File (filename)));
	 * 
	 * DataSetIterator Iter = new RecordReaderDataSetIterator(trainReader,
	 * batchSize, 2,1, true); RecordReader rr = new CSVRecordReader();
	 * rr.initialize(new FileSplit(new File(filename))); DataSetIterator Iter = new
	 * RecordReaderDataSetIterator.Builder(rr,batchSize)
	 * 
	 * .build(); return Iter;
	 * 
	 * }
	 */

	public static DataSetIterator generateDataSet(String filename, int batchSize) throws FileNotFoundException {

		FileReader file = new FileReader(filename);
		Scanner scanner = new Scanner(file);
		int count = 0;
		while (scanner.hasNextLine()) {
			scanner.nextLine();
			count++;
		}
		scanner.close();
		System.out.println("count" + count);
		String line = "";
		double[] res;
		double[][] inputs = new double[count][];
		double[] output = new double[count];
		int l = 0;
		// try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
			while ((line = br.readLine()) != null) {
				String[] country = line.split(",");
				// System.out.println("Createfile");

				res = new double[country.length - 1];
				for (int i = 0; i < country.length - 1; i++) {
					res[i] = Double.parseDouble(country[i]);
					inputs[l] = res;
				}
				output[l] = Double.parseDouble(country[country.length - 1]);

				l++;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		double[][] inputt = new double[inputs[0].length][inputs.length];
		inputt = transposeMatrix(inputs);
		System.out.println(" lines" + inputt.length);
		INDArray[] inputArray = new INDArray[inputt.length];
		for (int i = 0; i < inputArray.length; i++)
			inputArray[i] = Nd4j.create(inputt[i], new int[] { count, 1 });

		// INDArray inputNDArray = Nd4j.hstack(inputNDArray1,inputNDArray2);
		INDArray inputNDArray = Nd4j.hstack(inputArray);

		// System.out.println("NbrInput"+ inputNDArray .length());

		INDArray outPut = Nd4j.create(output, new int[] { count, 1 });
		DataSet dataSet = new DataSet(inputNDArray, outPut);
		// List<DataSet> listDs = dataSet.asList();
		java.util.List<DataSet> listDs = dataSet.asList();
		Collections.shuffle(listDs, rng);
		return new ListDataSetIterator(listDs, batchSize);
	}

	public MultiLayerNetwork getDeepDenseLayerNetworkConfiguration() {

		net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder().seed(seed).weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				// .updater(new org.nd4j.linalg.learning.config.Nesterovs(learningRate,0.9))

				.list()
				/*
				 * .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
				 * .activation(Activation.SIGMOID).build()) .layer(1, new
				 * DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
				 * .activation(Activation.SIGMOID).build())
				 * 
				 * .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
				 * .activation(Activation.IDENTITY)
				 * .nIn(numHiddenNodes).nOut(numOutputs).build())
				 */
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation(Activation.SIGMOID)
								.build())
				.layer(1,
						new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).activation(Activation.SIGMOID)
								.build())

				.layer(2,
						new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).activation(Activation.SIGMOID)
								.build())

				.layer(3,
						new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
								.nIn(numHiddenNodes).nOut(numOutputs).build())

				/*
				 * .layer(0, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
				 * .activation(Activation.TANH).build()) .layer(1, new
				 * OutputLayer.Builder(LossFunctions.LossFunction.MSE)
				 * .activation(Activation.IDENTITY)
				 * .nIn(numHiddenNodes).nOut(numOutputs).build())
				 */
				.pretrain(false).backprop(true).build());
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		return net;
	}

	public MultiLayerNetwork getDeepDenseLayerNetworkConfigurationEarlyStopping() {

		MultiLayerConfiguration myNetworkConfiguration = new NeuralNetConfiguration.Builder().seed(seed)
				.weightInit(WeightInit.XAVIER)
				// .updater(new Nesterovs(learningRate, 0.9))
				// .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				// .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
				.optimizationAlgo(OptimizationAlgorithm.LBFGS)
				// .updater(new org.nd4j.linalg.learning.config.Nesterovs(learningRate,0.9))
				// .updater(new Nesterovs(learningRate,0.9))
				// .updater(new org.nd4j.linalg.learning.config.Nesterovs(learningRate,0.9))
				.list()

				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation(Activation.SIGMOID)
								.build())
				.layer(1,
						new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).activation(Activation.SIGMOID)
								.build())
				.layer(2,
						new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).activation(Activation.SIGMOID)
								.build())
				.layer(3,
						new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).activation(Activation.SIGMOID)
								.build())
				.layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
						.nIn(numHiddenNodes).nOut(numOutputs).build())
				.pretrain(false).backprop(true).build();
		// net = new MultiLayerNetwork(
		EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
				.epochTerminationConditions(new MaxEpochsTerminationCondition(30))
				.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
				.scoreCalculator(new DataSetLossCalculator(iterTest, true)).evaluateEveryNEpochs(1)
				// .modelSaver(new LocalFileModelSaver(directory))
				.build();

		EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, myNetworkConfiguration, iterTrain);

		// Conduct early stopping training:
		// EarlyStoppingResult result = trainer.fit();
		result = trainer.fit();

		System.out.println("Termination reason: " + result.getTerminationReason());
		System.out.println("Termination details: " + result.getTerminationDetails());
		System.out.println("Total epochs: " + result.getTotalEpochs());
		System.out.println("Best epoch number: " + result.getBestModelEpoch());
		System.out.println("Score at best epoch: " + result.getBestModelScore());
		net = (MultiLayerNetwork) result.getBestModel();
		/*
		 * net.init(); net.setListeners(new ScoreIterationListener(1));
		 */

		return net;
	}

	public MultiLayerNetwork trainingTesting(int batchSize) throws IOException, InterruptedException {
		/*
		 * iterTrain = generateIterator( fileTrain, batchSize); iterTest =
		 * generateIterator( fileTest, batchSize);
		 */
		iterTrain = generateDataSet(fileTrain, batchSize);
		iterTest = generateDataSet(fileTest, batchSize);
		net = getDeepDenseLayerNetworkConfigurationEarlyStopping();
		nEpochs = result.getBestModelEpoch();
		/*
		 * net = getDeepDenseLayerNetworkConfiguration( ) ; for( int i=0; i<nEpochs; i++
		 * ){ iterTrain.reset(); net.fit(iterTrain); }
		 */
		for (int i = 0; i < nEpochs; i++) {
			iterTrain.reset();
			net.fit(iterTrain);
		}

		RegressionEvaluation eval = net.evaluateRegression(iterTest);
		System.out.println(eval.stats());
		return net;
	}

	public double prediction(double[] chain) throws FileNotFoundException {
		// net = trainingTesting ();
		final INDArray input = Nd4j.create(chain, new int[] { 1, chain.length });
		INDArray out = net.output(input, false);
		// System.out.println("prediction"+out.getDouble(0));
		return out.getDouble(0);
	}

}
