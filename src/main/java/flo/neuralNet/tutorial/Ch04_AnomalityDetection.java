package flo.neuralNet.tutorial;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;

import java.util.*;

import javax.imageio.ImageIO;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
public class Ch04_AnomalityDetection {
	
	public static String encodeArrayToImage(INDArray arr) throws IOException {
		BufferedImage bi = new BufferedImage(28,28,BufferedImage.TYPE_BYTE_GRAY);
		for(int i = 0; i < 784; i++) {
			bi.getRaster().setSample(i % 28, i/28, 0, (int)(255 * arr.getDouble(i)));
		}
		
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		ImageIO.write(bi, "PNG", baos);
		byte[] image = baos.toByteArray();
		baos.close();
		
		return Base64.getEncoder().encodeToString(image);
	}
	

	

	public static void main(String[] args) throws IOException {
		int seed = 12345;
		double lRate = 0.05;
		int numIn0 = 784;
		int numIn1 = 250;
		int numIn2 = 10;
		int numIn3 = 250;
		int numOut = 784;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.weightInit(WeightInit.XAVIER)
				.activation(Activation.RELU)
				.updater(new AdaGrad(lRate))
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.l2(0.0001)
				.list()
				.layer(0,new DenseLayer.Builder().nIn(numIn0).nOut(numIn1).build())
				.layer(1, new DenseLayer.Builder().nIn(numIn1).nOut(numIn2).build())
				.layer(2, new DenseLayer.Builder().nIn(numIn2).nOut(numIn3).build())
				.layer(3, new OutputLayer.Builder().nIn(numIn3).nOut(numOut).lossFunction(LossFunction.MSE).build())
				.pretrain(false).backprop(true)
				.build();
		
		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		network.setListeners(new ScoreIterationListener(2));
		
		
		//Load data (50,000). Use it as 40,000 for training and 10,000 for test
		int batchSize = 100;
		int numExamples = 50000;
		MnistDataSetIterator iter = new MnistDataSetIterator(batchSize,numExamples,false);
	
		 ArrayList<INDArray> featuresTrain = new ArrayList<INDArray>();
		 ArrayList<INDArray> featuresTest = new ArrayList<INDArray>(); 
		 ArrayList<INDArray> labelsTest = new ArrayList<INDArray>();
		 
		 Random rand = new Random(seed);
//		 RandomStream rand = new MRG32k3a();
		 
		 while(iter.hasNext()) {
			 DataSet data = iter.next();
			 SplitTestAndTrain split = data.splitTestAndTrain(80,rand); // 80 of the 100(batchSize) for training
			 featuresTrain.add(split.getTrain().getFeatures());
			 featuresTest.add(split.getTest().getFeatures());
			 INDArray indices = Nd4j.argMax(split.getTest().getLabels(), 1);
			 labelsTest.add(indices);
		 }
		 
		 int numEpochs = 30;
		 for(int e = 0; e < numEpochs; e++) {
			 for(INDArray data : featuresTrain)
				 network.fit(data,data);
			 System.out.println("Epoch " + e + " complete");
		 }
		 
		//Evaluate the model on the test data
		//Score each example in the test set separately
		//Compose a map that relates each digit to a list of (score, example) pairs
		//Then find N best and N worst scores per digit
		 
		 HashMap<Integer, List<Pair<Double, INDArray>>> listsByDigit = new HashMap<Integer,List<Pair<Double,INDArray>>>();
		
		 for(int i = 0; i < 10; i++) {
			 listsByDigit.put(i, new ArrayList<Pair<Double, INDArray>>());
		 }
		 for(int i = 0; i < featuresTest.size(); i++) {
			 INDArray testData = featuresTest.get(i);
			 INDArray labels = labelsTest.get(i);
			 
			 for(int j = 0; j < testData.rows(); j++) {
				 INDArray example = testData.getRow(j);
				int digit = (int)labels.getDouble(j);
				double score = network.score(new DataSet(example,example));
				//Add (score,example pairs)
				List<Pair<Double, INDArray>> digitAllPairs = listsByDigit.get(digit);
				digitAllPairs.add(new ImmutablePair<Double,INDArray>(score,example));
			 }
		 }
		 
		 //Sort list by score
		 Comparator<Pair<Double, INDArray>> c = new Comparator<Pair<Double,INDArray>>(){
			public int compare(Pair<Double, INDArray> o1, Pair<Double, INDArray> o2) {
				return Double.compare(o1.getLeft(), o2.getLeft());
			}
		 };
		 
		 for(List<Pair<Double, INDArray>> digitAllPairs : listsByDigit.values()) {
			 Collections.sort(digitAllPairs, c);
		 }
		 
		 ArrayList<INDArray> best = new ArrayList<INDArray>(50); //50 best
		 ArrayList<INDArray> worst = new ArrayList<INDArray>(50); //50 worst
		 
		 for (int i = 0; i < 10; i++) {
			 List<Pair<Double, INDArray>> list = listsByDigit.get(i);
			 for(int j = 0; j < 5; j++) {
				 best.add(list.get(j).getRight());
				 worst.add(list.get(list.size()-j-1).getRight());
			 }
				 
		 }
		 
		 
		 
		 
	}
}
