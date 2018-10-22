package flo.lacZ;

import umontreal.ssj.markovchainrqmc.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import flo.neuralNet.NeuralNet;
import umontreal.ssj.hups.*;
import umontreal.ssj.rng.*;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.*;
import umontreal.ssj.util.sort.*;

public class cAMPTestNN extends ArrayOfComparableChainsNN<cAMPComparable> {

	public cAMPTestNN (cAMPComparable baseChain, String[] fileNames) {
		super(baseChain, fileNames);
        //  this.baseChain = baseChain;
	}

	// Applies Array-RQMC for Schloglsystem with various point sets and given sort, m
	// times independently.
	// sortCoord is the number of coordinates used to sort the points. 
	
	// dimState is the dimension of the state that is used for the sort.
	// The points are sorted based on the first dimState coordinates.
	// numSets is the number of point sets for which the experiment is made.
	//
	public void testMethods (cAMPComparable biology, int sortCoordPts, int numSteps, int m, int numSets, String sor) throws IOException {
		int s;   // Index of point set.
		int[] N = {  512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
				131072, 262144, 524288, 1048576 }; // logn from 8 to 20.
		int[] a = {  115, 851, 1553, 2839, 6685, 9945, 12421, 38295, 114789,
				177473, 286857, 271251 };
		PointSet[] pointSets = new PointSet[numSets];
		CachedPointSet[] pointSetsNUS = new CachedPointSet[numSets];
		PointSetRandomization randShift = new RandomShift(new MRG32k3a());
		PointSetRandomization randDigital = new LMScrambleShift(new MRG32k3a());
		PointSetRandomization randNUS = new NestedUniformScrambling (new MRG32k3a(), 31);  // Scramble all 31 bits.
		DigitalNetBase2 sobolNet;
		// Monte Carlo experiments with nMC independent runs.
		int nMC = 1000 * 1000;   // n to estimate MC variance.
		Tally statMC = new Tally();
		statMC.init();
		biology.simulRunsWithSubstreams(nMC, biology.T, new MRG32k3a(), statMC);
		double varMC = statMC.variance();
		System.out.println("\n\n --------------------------");
		System.out.println("MC average  = " + statMC.average());
		System.out.println("MC variance = " + varMC);
		
		
	
		

		

		
		    
		 
		  // Independent points (Monte Carlo) 
		for (s = 0; s < numSets; ++s) {
		  pointSets[s] = new IndependentPointsCached(N[s], baseChain.N+baseChain.K); }
		  System.out.println (testVarianceRateFormat (pointSets, randShift, 0, 
					 numSteps, m, varMC, "cAMP+Independent points"+sor, "Independent points")); 
		  
	
			
		// Stratification
		int k;   
	    for (s = 0; s < numSets; ++s) {
			k = (int) Math.round(Math.pow(Num.TWOEXP[s + 9],
				 1.0 / (double) (sortCoordPts + baseChain.K)));
			pointSets[s] = new StratifiedUnitCube (k, sortCoordPts +baseChain.K);
			// Here the points must be sorted at each step, always.
			// In the case of Hilbert map, the points should be 2d and sorted based on one coordinate,
			// whereas the states are 2d and sorted by the Hilbert sort.
	    }
		System.out.println (testVarianceRateFormat (pointSets, randShift, sortCoordPts, 
				 numSteps, m, varMC, "cAMP+Stratification"+sor, "Stratification")); 


		// Sobol
		for (s = 0; s < numSets; ++s) {
		
			
		}
		// When the point set is sorted only one, it must be sorted before calling this function.
		System.out.println (testVarianceRateFormat (pointSets, randDigital, 0, 
			 numSteps, m, varMC, "cAMP+Sobol+LMS"+sor, "Sobol + LMS")); 

		
		// Sobol + LMS+  Baker
		for (s = 0; s < numSets; ++s) {
			
			pointSets[s] = new BakerTransformedPointSet (new SobolSequence(s + 9,
					31, 1+baseChain.K));
		
		}
	
		System.out.println (testVarianceRateFormat (pointSets, randDigital, 0, 
				 numSteps, m, varMC, "cAMP+Sobol+LMS+Baker"+sor, "Sobol + LMS+ Baker")); 
		
		
		// Sobol NUS
		for (s = 0; s < numSets; ++s) {
			
				CachedPointSet p = new CachedPointSet(new SobolSequence(s + 9, 31, 1+baseChain.K));
				p.setRandomizeParent(false);
				pointSets[s] = p;
			
		}
		// When the point set is sorted only one, it must be sorted before calling this function.
		System.out.println (testVarianceRateFormat (pointSets, randNUS, 0,
			 numSteps, m, varMC, "cAMP+Sobol+NUS"+sor, "Sobol + NUS\n"));
		
		//Korobov 
		
		 for (s = 0; s < numSets; ++s){
			
				pointSets[s] = new KorobovLattice(N[s],a[s],1+baseChain.K, 1);
			
		 }
		 
		  System.out.println (testVarianceRateFormat (pointSets, randShift, 0,  
					 numSteps, m, varMC, "cAMP+Korobov"+sor, 
					 //  "Asian-var-korobov-Baker", 
					 "Korobov "));
		// Korobov lattice
		for (s = 0; s < numSets; ++s) {
			
				pointSets[s] = new BakerTransformedPointSet (new KorobovLattice
						(N[s], a[s], 1+baseChain.K, 1));
			
		}
	    System.out.println (testVarianceRateFormat (pointSets, randShift, 0,  
				 numSteps, m, varMC, "cAMP+Korobov+Baker"+sor, 
				 //  "Asian-var-korobov-Baker", 
				 "Korobov + Baker"));
		}
	
	public static void usage() {
		System.err.println("usage: SchloglsystemTest { split | batch | hilbertbatch | hilbert { normal | logistic <x0> <w> | logisticGC <mu> <c> } }");
		System.err.println("examples:");
		System.err.println("    SchloglsystemTest split");
		System.err.println("    SchloglsystemTest batch");
		System.err.println("    SchloglsystemTest hilbert normal");
		System.err.println("    SchloglsystemTest hilbert logistic 0.0 2.0");
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		
		// int numSteps = d;

		int m = 100; // Number of replications.
		int numSets = 12; // Number of point sets to try.
		int K=6;
	
		//double[] c={ 3* Math.pow(10, -7), Math.pow(10, -4), Math.pow(10, -3), 3.5};
		double[] c={ 8.696*Math.pow(10, -20),  0.02, 1.154*Math.pow(10, -19), 0.02, 0.016, 0.0017*Math.pow(10, -15) };
		
		
		
		double []  X0={30, 30, 10, 10, 10, 10};
		int[][] Reactant={{0,1},
				          {2},
				          {2,1},
				          {3},
				          {3},
				          {4,5},
				         
		};
		int[][] nuReactant={{1,2},
		          {1},
		          {1,2},
		          {1},
		          {1},
		          {1,2},
		         
};
       int [][] Product={{2},
		          {0,1},
		          {3},
		          {2,1},
		          {4,5},
		          {3},
		        
};
       int [][] nuProduct={{1},
		          {1,2},
		          {1},
		          {1,2},
		          {1,2},
		          {1},
		         
};
		
		double  tau=0.05;
		
		int T = 10;
		int numSteps =T;
	
		int N =6;
		cAMPComparable.RealsTo01Map map = null;
		//double epsilon = Math.pow(10, -4);
		double epsilon = 0.1;
		int[] hor={2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1};
		int[] nuHor={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		
		/*if (args.length < 1) {
			usage();
			System.exit(1);
		}

		String sortType = args[0];

		if (sortType.equals("hilbert")) {

			if (args.length == 2 && args[1].equals("normal")) {
				System.out.println("Using normal CDF map");
			}
			else if (args.length == 4 && args[1].equals("logisticGC")) {
				double mu = Double.parseDouble(args[2]);
				double f = Double.parseDouble(args[3]);
				System.out.println("Using Gerber & Chopin logistic map centered at " + mu + " with half width " + f);
			}
			else if (args.length == 4 && args[1].equals("logistic")) {
				double x0 = Double.parseDouble(args[2]);
				double w = Double.parseDouble(args[3]);
				System.out.println("Using logistic map centered at " + x0 + " with scale " + w);
			}
			else {
				usage();
				System.exit(1);
			}
		}*/
		//ProkaryoticComparable biology=  new ProkaryoticComparable(K, N, c, X0 , S, tau, T, map);
		cAMPComparable biology=  new cAMPComparable(K, N, c, X0 , tau, Reactant, Product, nuReactant ,  nuProduct, T, map, epsilon);
		//ProkaryoticComparableold biology=  new ProkaryoticComparableold(K, N, c, X0 , S, tau, T, map);
		
		//cAMPTestNN test = new cAMPTestNN (biology);    // This is the array of comparable chains.
		// PointSetRandomization rand = new RandomShift(new MRG32k3a());

		MultiDimSort<cAMPComparable> sort;
		MultiDimSort01<MultiDim01>  sort01;
		MultiDimSortN<MultiDim> sort02;
		double[] batchExp = { 0.2, 0.1, 0.2, 0.1, 0.2, 0.2 };
		
		System.out.println(biology.toString());
		
		
	
		
	/*	if (sortType.equals("split")) {
			System.out.println("\n******************************************\n");
			System.out.println("\nsort type: SPLIT SORT\n");
			sort = new SplitSort<ProkaryoticComparable> (3);
			test.testMethods (biology, sort, 3,T, m, numSets);
		}
		else if (sortType.equals("batch")) {
			System.out.println("\n******************************************");
			System.out.println("\nsort type: BATCH SORT\n");
			sort = new BatchSort<ProkaryoticComparable>(batchExp);    
			test.testMethods(biology, sort, 3, T, m, numSets);
		}
		else if (sortType.equals("hilbertbatch")) {
			System.out.println("\n******************************************");
			System.out.println("\nsort type: HILBERT BATCH SORT\n");
			sort = new HilbertCurveBatchSort<ProkaryoticComparable>(batchExp, 20);
			test.testMethods(biology, sort, 1, T, m, numSets);
		}
		else if (sortType.equals("hilbert")) {
			System.out.println("\n******************************************");
			System.out.println("\nsort type: HILBERT SORT\n");
			sort01 = new HilbertCurveSort(3, 12);
			test.testMethods(biology, sort01, 1, T, m, numSets);
		}
		else {
			System.out.println("invalid sort type!");
			usage();
			System.exit(1);
		}

		System.out.println("\n  Done !!!");
	}*/


		
		 
		/* System.out.println("\n *************  SPLIT SORT  *************** \n");
		 sort = new SplitSort (6);		
		 test.testMethods (biology, sort, 6, T, m, numSets, "Split");

		System.out.println("\n *************  BATCH SORT  *************** \n");
		sort = new BatchSort<cAMPComparable>(batchExp);    // Sort in 3 dim.
		test.testMethods(biology, sort, 6, T, m, numSets, "Batch");
		System.out.println("\n **********  HILBERT BATCH SORT  ***********\n");
		sort = new HilbertCurveBatchSort<cAMPComparable>(batchExp, 20);   
		test.testMethods(biology, sort, 1, T, m, numSets, "HilbertBatch");

		System.out.println("\n *************  HILBERT SORT  *************** \n");
		sort01 = new HilbertCurveSort(1, 12);
		test.testMethods (biology, sort01, 1, T, m, numSets, "Hilbert");*/
		
		NeuralNet testN = new NeuralNet(biology,"data/cAMP/"); 
		boolean genData = true;
		int numChains = 524288 * 2;
		int logNumChains = 19 + 1;

		Chrono timer = new Chrono();
		RandomStream stream = new MRG32k3a();
		String dataLabel = "MCData";
//		PointSet sobol = new SobolSequence(logNumChains, 31, numSteps * 24);
//		PointSetRandomization rand = new LMScrambleShift(stream);
//		RQMCPointSet p = new RQMCPointSet(sobol, rand);

		if (genData) {
			timer.init();
			testN.genData(dataLabel, numChains, numSteps,stream);
			System.out.println("\n\nTiming:\t" + timer.format());
		}
		/*
		 ***********************************************************************
		 ************* NEURAL NETWORK*******************************************
		 ***********************************************************************
		 */

		int currentStep = 1;

		int batchSize = 128;
		//int batchSize = numChains/8;
		int numEpochs = 32;

		/*
		 * READ DATA
		 */

		ArrayList<DataSet> dataAllList = new ArrayList<DataSet>();
		for(int s = 0; s < numSteps; s++) {
			dataAllList.add(testN.getData(dataLabel,s,numChains));
		}

		//*
		 /* GENERATE NETWORKS
		 */
		double lRate = 0.1;
		ArrayList<MultiLayerNetwork> networkList = new ArrayList<MultiLayerNetwork>();
		for (int i = 0; i < numSteps; i++) {
//			lRate += 1.0;
			networkList.add(testN.genNetwork(i, 6, lRate));
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
		NeuralNet NN = new NeuralNet(biology,"data/cAMP/");
		for(int i = 0; i < numSteps; i ++) {
			
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

			//NeuralNet.trainNetwork(network, trainingData, numEpochs, batchSize, (numChains / batchSize) * 1, 1000);
			NN.trainNetwork(network, trainingData, numEpochs, batchSize, (numChains / batchSize) * 1, 1000);
			// TEST NETWORK
			str = NN.testNetwork(network, testData, batchSize);
			sb.append(str);
			System.out.println(str);


			// saveNetwork(network,"Asian_Step" + currentStep, normalizer);
			testN.saveNetwork(network, "cAMP_Step" + i, normalizer);
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

	
		
		String [] fileNames = new String[numSteps];
		String base = "data/cAMP/";
		for(int j = 0; j < numSteps; j++) {
			fileNames[j] = base + "cAMP_Step" + j + ".zip";
		}
		
		cAMPTestNN test = new cAMPTestNN (biology, fileNames);  

		System.out.println("\n *************  NeuralNetwork  SORT  *************** \n");
//		sort02 = new NeuralNetworkSort(2);
		// test.testMethods (biology, sort02, 1, T, m, numSets, "NeuralNetworkSort");
		// sort02 = new NeuralNetworkSort(2);
		test.testMethods(biology, 1, numSteps, m, numSets, "NeuralNetwork");
	}
}
