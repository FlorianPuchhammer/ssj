package tau_leaping2;

import umontreal.ssj.markovchainrqmc.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import umontreal.ssj.hups.*;
import umontreal.ssj.rng.*;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.*;
import umontreal.ssj.util.sort.*;

public class LacZLacYsystemwithoutnegativepopulationTestNN extends ArrayOfComparableChainsNN<LacZLacYwithoutnegativepopulationComparable> {

	/*public LacZLacYsystemwithoutnegativepopulationTestNN (LacZLacYwithoutnegativepopulationComparable baseChain) {
		super(baseChain);
        //  this.baseChain = baseChain;
	}*/
	public LacZLacYsystemwithoutnegativepopulationTestNN (LacZLacYwithoutnegativepopulationComparable baseChain, String[] fileNames) {
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
	public void testMethods (LacZLacYwithoutnegativepopulationComparable biology, 
	        MultiDimSort sort, int sortCoordPts, int numSteps, int m, int numSets) {
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
		int nMC = 1048576;   // n to estimate MC variance.
		Tally statMC = new Tally();
		statMC.init();
		biology.simulRunsWithSubstreams(nMC, biology.T, new MRG32k3a(), statMC);
		double varMC = statMC.variance();
		System.out.println("\n\n --------------------------");
		System.out.println("MC average  = " + statMC.average());
		System.out.println("MC variance = " + varMC);
		
		
	
		

		

		
		    
		 /*
		  // Independent points (Monte Carlo) 
		for (s = 0; s < numSets; ++s) {
		  pointSets[s] = new IndependentPointsCached(N[s], baseChain.N+2+baseChain.K); }
		  System.out.println (testVarianceRateFormat (pointSets, randShift, sort, 0, 
					 numSteps, m, varMC, "LacYLacZ+Independent points"+sort.toString(), "Independent points")); 
		  
	
			
		// Stratification
		int k;   
	    for (s = 0; s < numSets; ++s) {
			k = (int) Math.round(Math.pow(Num.TWOEXP[s + 9],
				 1.0 / (double) (sortCoordPts + baseChain.K)));
			pointSets[s] = new StratifiedUnitCube (k, sortCoordPts+2 +baseChain.K);
			// Here the points must be sorted at each step, always.
			// In the case of Hilbert map, the points should be 2d and sorted based on one coordinate,
			// whereas the states are 2d and sorted by the Hilbert sort.
	    }
		System.out.println (testVarianceRateFormat (pointSets, randShift, sort, sortCoordPts, 
				 numSteps, m, varMC, "LacYLacZ+Stratification"+sort.toString(), "Stratification")); */


		// Sobol
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new SobolSequence(s + 9, 31, 1+baseChain.K+2);
			else
				// The points are sorted here, but only once.
			    pointSets[s] = new SortedAndCutPointSet (new SobolSequence(s + 9, 31, sortCoordPts+baseChain.K+2), sort);
		}
		// When the point set is sorted only one, it must be sorted before calling this function.
		System.out.println (testVarianceRateFormat (pointSets, randDigital, sort, 0, 
			 numSteps, m, varMC, "LacYLacZ+Sobol+LMS"+sort.toString(), "Sobol + LMS")); 

		/*
		// Sobol + LMS+  Baker
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
			pointSets[s] = new BakerTransformedPointSet (new SobolSequence(s + 9,
					31, 1+baseChain.K+2));
			else
				 pointSets[s] = new SortedAndCutPointSet (new BakerTransformedPointSet (new SobolSequence(s + 9,
							31, sortCoordPts+baseChain.K+2)), sort);
		}
	
		System.out.println (testVarianceRateFormat (pointSets, randDigital, sort, 0, 
				 numSteps, m, varMC, "LacYLacZ+Sobol+LMS+Baker"+sort.toString(), "Sobol + LMS+ Baker")); 
		
		
		// Sobol NUS
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1) {
				CachedPointSet p = new CachedPointSet(new SobolSequence(s + 9, 31, 1+baseChain.K+2));
				p.setRandomizeParent(false);
				pointSets[s] = p;
			}
			else {
				CachedPointSet p = new CachedPointSet(new SobolSequence(s + 9, 31, sortCoordPts+baseChain.K+2));
				p.setRandomizeParent(false);
				// The points are sorted here, but only once.
			    pointSets[s] = new SortedAndCutPointSet (p, sort);
			 }
		}
		// When the point set is sorted only one, it must be sorted before calling this function.
		System.out.println (testVarianceRateFormat (pointSets, randNUS, sort, 0,
			 numSteps, m, varMC, "LacYLacZ+Sobol+NUS"+sort.toString(), "Sobol + NUS\n"));
		
		//Korobov 
		
		 for (s = 0; s < numSets; ++s){
			 if (sortCoordPts == 1)		 
				pointSets[s] = new KorobovLattice(N[s],a[s],1+baseChain.K+2, 1);
			 else
				pointSets[s] = new SortedAndCutPointSet (new KorobovLattice(N[s],a[s],sortCoordPts+baseChain.K+2, 1),sort);
		 }
		 
		  System.out.println (testVarianceRateFormat (pointSets, randShift, sort, 0,  
					 numSteps, m, varMC, "LacYLacZ+Korobov"+sort.toString(), 
					 //  "Asian-var-korobov-Baker", 
					 "Korobov "));
		// Korobov lattice
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new BakerTransformedPointSet (new KorobovLattice
						(N[s], a[s], 1+baseChain.K+2, 1));
			else
				// The points are sorted here, but only once.
			    pointSets[s] = new SortedAndCutPointSet (new BakerTransformedPointSet 
			    		(new KorobovLattice (N[s], a[s], sortCoordPts+baseChain.K+2, 1)), sort);
		}
	    System.out.println (testVarianceRateFormat (pointSets, randShift, sort, 0,  
				 numSteps, m, varMC, "LacYLacZ+Korobov+Baker"+sort.toString(), 
				 //  "Asian-var-korobov-Baker", 
				 "Korobov + Baker"));
		*/
	}
	public void testMethods (LacZLacYwithoutnegativepopulationComparable biology, 
	        int sortCoordPts, int numSteps, int m, int numSets, String sort) throws IOException {
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
		int nMC = 1048576;   // n to estimate MC variance.
		Tally statMC = new Tally();
		statMC.init();
		biology.simulRunsWithSubstreams(nMC, biology.T, new MRG32k3a(), statMC);
		double varMC = statMC.variance();
		System.out.println("\n\n --------------------------");
		System.out.println("MC average  = " + statMC.average());
		System.out.println("MC variance = " + varMC);
		
		
	
		

		

		
		    
		 /*
		  // Independent points (Monte Carlo) 
		for (s = 0; s < numSets; ++s) {
		  pointSets[s] = new IndependentPointsCached(N[s], baseChain.N+2+baseChain.K); }
		  System.out.println (testVarianceRateFormat (pointSets, randShift, sort, 0, 
					 numSteps, m, varMC, "LacYLacZ+Independent points"+sort, "Independent points")); 
		  
	
			
		// Stratification
		int k;   
	    for (s = 0; s < numSets; ++s) {
			k = (int) Math.round(Math.pow(Num.TWOEXP[s + 9],
				 1.0 / (double) (sortCoordPts + baseChain.K)));
			pointSets[s] = new StratifiedUnitCube (k, sortCoordPts+2 +baseChain.K);
			// Here the points must be sorted at each step, always.
			// In the case of Hilbert map, the points should be 2d and sorted based on one coordinate,
			// whereas the states are 2d and sorted by the Hilbert sort.
	    }
		System.out.println (testVarianceRateFormat (pointSets, randShift, sort, sortCoordPts, 
				 numSteps, m, varMC, "LacYLacZ+Stratification"+sort, "Stratification")); */


		// Sobol
		for (s = 0; s < numSets; ++s) {
			
				pointSets[s] = new SobolSequence(s + 9, 31, 1+baseChain.K+2);
		
		}
		// When the point set is sorted only one, it must be sorted before calling this function.
		System.out.println (testVarianceRateFormat (pointSets, randDigital, 0, 
			 numSteps, m, varMC, "LacYLacZ+Sobol+LMS"+sort, "Sobol + LMS")); 

		/*
		// Sobol + LMS+  Baker
		for (s = 0; s < numSets; ++s) {
			
			pointSets[s] = new BakerTransformedPointSet (new SobolSequence(s + 9,
					31, 1+baseChain.K+2));
			
		}
	
		System.out.println (testVarianceRateFormat (pointSets, randDigital, sort, 0, 
				 numSteps, m, varMC, "LacYLacZ+Sobol+LMS+Baker"+sort, "Sobol + LMS+ Baker")); 
		
		
		// Sobol NUS
		for (s = 0; s < numSets; ++s) {
			
				CachedPointSet p = new CachedPointSet(new SobolSequence(s + 9, 31, 1+baseChain.K+2));
				p.setRandomizeParent(false);
				pointSets[s] = p;
			
		}
		// When the point set is sorted only one, it must be sorted before calling this function.
		System.out.println (testVarianceRateFormat (pointSets, randNUS, sort, 0,
			 numSteps, m, varMC, "LacYLacZ+Sobol+NUS"+sort, "Sobol + NUS\n"));
		
		//Korobov 
		
		 for (s = 0; s < numSets; ++s){
			
				pointSets[s] = new KorobovLattice(N[s],a[s],1+baseChain.K+2, 1);
			
		 }
		 
		  System.out.println (testVarianceRateFormat (pointSets, randShift, sort, 0,  
					 numSteps, m, varMC, "LacYLacZ+Korobov"+sort, 
					 //  "Asian-var-korobov-Baker", 
					 "Korobov "));
		// Korobov lattice
		for (s = 0; s < numSets; ++s) {
			
				pointSets[s] = new BakerTransformedPointSet (new KorobovLattice
						(N[s], a[s], 1+baseChain.K+2, 1));
			
		}
	    System.out.println (testVarianceRateFormat (pointSets, randShift, sort, 0,  
				 numSteps, m, varMC, "LacYLacZ+Korobov+Baker"+sort, 
				 //  "Asian-var-korobov-Baker", 
				 "Korobov + Baker"));
		*/
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
		int K=22;
	
		//double[] c={ 3* Math.pow(10, -7), Math.pow(10, -4), Math.pow(10, -3), 3.5};
		double[] c={ 0.17, 10, 1, 1, 0.015, 1, 0.36, 0.17, 0.45, 0.17, 0.45, 0.4, 0.4, 0.015, 0.036, 6.42* Math.pow(10, -5), 6.42* Math.pow(10, -5), 0.3, 0.3, 9.52* Math.pow(10, -5), 431, 14};
		
		 double genTime                  = 2100;
		//double []  X0={1, 0, 0,  0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0};
		
		double []  X0={100, 50, 50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50};
		double[][] S={{ -1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0, 0,0,0}, 
				{ -1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0, 0,0,0}, 
				{ 1, -1, -1,0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0, 0,0,0}, 
				{ 0, 0, 1,-1 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0, 0,0,0}, 
				{ 0, 0, 0,1 , 0, 0, 0, -1, 1, 0, 0, 1, 0, 0, 0, 0, 0,-1 ,0, 0,0,0}, 
				{ 0, 0, 0,1 , -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0, 0,0,0}, 
				{ 0, 0, 0,0 , 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0, 0,0,0},
				{ 0, 0, 0,0 , 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0,0 ,-1, 0,0,0},
				{ 0, 0, 0,0 , 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0, 0,0,0},
				{ 0, 0, 0,0 , 0, 0, 0, -1, 1, -1, 1, 0, 0, 0, 0, 0, 0,0 ,0, 0,0,0},
				{ 0, 0, 0,0 , 0, 0, 0, 1, -1, 0, 0, -1, 0, 0, 0, 0, 0,0 ,0, 0,0,0},
				{ 0, 0, 0,0 , 0, 0, 0, 0, 0, 1, -1, 0, -1, 0, 0, 0, 0,0 ,0, 0,0,0},
				{ 0, 0, 0,0 , 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0,0 ,0, 0,0,0},
				{ 0, 0, 0,0 , 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0,0 ,0, 0,0,0},
				{ 0, 0, 0,0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0,0 ,0, -1,1,0},
				{ 0, 0, 0,0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1,0 ,0, 0,0,0},
				{0, 0, 0,  0, 0, 0, 0, 0, 0,0,0,0,0,0,0,1,0,0,0,0,0,0},
				{0, 0, 0,  0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,1,0,0,0,0,0},
				{0, 0, 0,  0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,1,0,0,0,0},
				{0, 0, 0,  0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,1,0,0,0},				
				{0, 0, 0,  0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,-1,0, 1},
				{0, 0, 0,  0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,1,-1, 0},
				{0, 0, 0,  0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,1, 0},
				};
		double  tau=0.2;
		
		int T = 10;
		int numSteps = T;
	
		int N =23;
		LacZLacYwithoutnegativepopulationComparable.RealsTo01Map map = null;
		//double epsilon = Math.pow(10, -4);
		double epsilon = 0.03;
		int[] hor={2, 2, 1, 1, 2, 1, 1, 2, 1,2,1,1,1,1,2,1,1,1,1,1,2,1,1};
		int[] nuHor={0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		
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
		
		int[][] Reactant={{0,1},
		          {2},
		          {2},
		          {3},
		          {5},
		          {6},
		          {8},
		          {4,9},
		          {10},
		          {7,9},
		          {11},		        	  
		          {10},
		          {11},
		          {12},
		          {13},
		          {14},
		          {15},
		          {4},
		          {7},
		          {14, 20},
		          {21},
		          {15},
};
		int[][] nuReactant={{1,1},
		          {1},
		          {1},
		          {1},
		          {1},
		          {1},
		          {1},
		          {1},
		          {1},
		          {1,1},
		          {1},		        	  
		          {1},
		          {1},
		          {1},
		          {1},
		          {1},
		          {1},
		          {1},
		          {1},
		          {1, 1},
		          {1},
		          {1},
};
		int [][] nuProduct={{1},
		        {1, 1},
		        {1},
		        {1,1,1},
		        {1},
		        {1,1},
		        {1},
		        {1},
		        {1,1},
		        {1},
		        {1,1},		        	  
		        {1,1},
		        {1,1},
		        {1},
		        {1},
		        {1},
		        {1},
		        {1},
		        {1},
		        {1},
		        {1,1},
		        {1,1},
		};
int [][] Product={{2},
        {0, 1},
        {3},
        {0,4,5},
        {6},
        {7,8},
        {1},
        {10},
        {4,9},
        {11},
        {7,9},		        	  
        {4,12},
        {7,13},
        {14},
        {15},
        {16},
        {17},
        {18},
        {19},
        {21},
        {22,14},
        {20,15},
};
		LacZLacYwithoutnegativepopulationComparable biology=  new LacZLacYwithoutnegativepopulationComparable(K, N, c, X0 , S, tau, Reactant, Product, nuReactant ,  nuProduct, T, map, epsilon);
		//LacZLacYwithoutnegativepopulationComparableold biology=  new LacZLacYwithoutnegativepopulationComparableold(K, N, c, X0 , S, tau, T, map);
		
		   // This is the array of comparable chains.
		// PointSetRandomization rand = new RandomShift(new MRG32k3a());

		MultiDimSort<LacZLacYwithoutnegativepopulationComparable> sort;
		MultiDimSort01<MultiDim01>  sort01;
		double[] batchExp = { 0.04, 0.04, 0.06, 0.06, 0.04, 0.04,0.06, 0.06, 0.04,0.04, 0.06, 0.06,0.04, 0.04, 0.06,0.06, 0.02, 0.02,0.03, 0.03, 0.02,0.02, 0.06 };
		
		System.out.println(biology.toString());
		
		
	
		
	/*	if (sortType.equals("split")) {
			System.out.println("\n******************************************\n");
			System.out.println("\nsort type: SPLIT SORT\n");
			sort = new SplitSort<LacZLacYwithoutnegativepopulationComparable> (3);
			test.testMethods (biology, sort, 3,T, m, numSets);
		}
		else if (sortType.equals("batch")) {
			System.out.println("\n******************************************");
			System.out.println("\nsort type: BATCH SORT\n");
			sort = new BatchSort<LacZLacYwithoutnegativepopulationComparable>(batchExp);    
			test.testMethods(biology, sort, 3, T, m, numSets);
		}
		else if (sortType.equals("hilbertbatch")) {
			System.out.println("\n******************************************");
			System.out.println("\nsort type: HILBERT BATCH SORT\n");
			sort = new HilbertCurveBatchSort<LacZLacYwithoutnegativepopulationComparable>(batchExp, 20);
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
		NeuralNet testN = new NeuralNet(biology); 
		boolean genData = true;
		int numChains = 524288 * 2;
		int logNumChains = 19 + 1;

		Chrono timer = new Chrono();
		RandomStream stream = new MRG32k3a();
		String dataLabel = "SobolData";
		PointSet sobol = new SobolSequence(logNumChains, 31, numSteps * 24);
		PointSetRandomization rand = new LMScrambleShift(stream);
		RQMCPointSet p = new RQMCPointSet(sobol, rand);

		if (genData) {
			timer.init();
			testN.genData(dataLabel, numChains, numSteps, p.iterator());
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
			networkList.add(testN.genNetwork(6, lRate));
		}
		
		/*
		 * TRAIN NETWORK
		 */
		FileWriter fw = new FileWriter("/u/benabama/SSJ2/WorkspaceFinal/array-rqmcFinal/data/comparisonLacYLaZ.txt");
		StringBuffer sb = new StringBuffer("");
		String str;

		DataSet dataAll, trainingData, testData;
		double ratioTrainingData = 0.8;
		DataNormalization normalizer;
		MultiLayerNetwork network;
		SplitTestAndTrain testAndTrain;
		NeuralNet NN = new NeuralNet(biology,"/u/benabama/SSJ2/WorkspaceFinal/array-rqmcFinal/data/LacYLaZ/");
		for(int i = 1; i < numSteps; i ++) {
			
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
			NeuralNet.trainNetwork(network, trainingData, numEpochs, batchSize, (numChains / batchSize) * 1, 1000);
			// TEST NETWORK
			str = NeuralNet.testNetwork(network, testData, batchSize);
			sb.append(str);
			System.out.println(str);


			// saveNetwork(network,"Asian_Step" + currentStep, normalizer);
			testN.saveNetwork(network, "LacYLaZ_Step" + i, normalizer);
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
		String base = "/u/benabama/SSJ2/WorkspaceFinal/array-rqmcFinal/data/LacYLaZ/";
		for(int j = 0; j < numSteps; j++) {
			fileNames[j] = base + "LacYLaZ_Step" + j + ".zip";
		}
		
		LacZLacYsystemwithoutnegativepopulationTestNN test = new LacZLacYsystemwithoutnegativepopulationTestNN (biology, fileNames);  

		System.out.println("\n *************  NeuralNetwork  SORT  *************** \n");
//		sort02 = new NeuralNetworkSort(2);
		// test.testMethods (biology, sort02, 1, T, m, numSets, "NeuralNetworkSort");
		// sort02 = new NeuralNetworkSort(2);
		test.testMethods(biology, 1, numSteps, m, numSets, "NeuralNetwork");
		
		/* 
		 System.out.println("\n *************  SPLIT SORT  *************** \n");
		 sort = new SplitSort (23);		
		 test.testMethods (biology, sort, 15, T, m, numSets);

		 	System.out.println("\n *************  BATCH SORT  *************** \n");
		sort = new BatchSort<LacZLacYwithoutnegativepopulationComparable>(batchExp);    // Sort in 3 dim.
		test.testMethods(biology, sort, 15, T, m, numSets);
		System.out.println("\n **********  HILBERT BATCH SORT  ***********\n");
		sort = new HilbertCurveBatchSort<LacZLacYwithoutnegativepopulationComparable>(batchExp, 20);   
		test.testMethods(biology, sort, 1, T, m, numSets);

		System.out
				.println("\n *************  HILBERT SORT  *************** \n");
		sort01 = new HilbertCurveSort(23, 12);
		test.testMethods (biology, sort01, 1, T, m, numSets);*/

		System.out.println("\n  Done !!!");
	}
}
