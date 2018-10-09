package flo.biologyArrayRQMC;

import umontreal.ssj.markovchainrqmc.*;

import java.io.IOException;

import umontreal.ssj.hups.*;
import umontreal.ssj.rng.*;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.*;
import umontreal.ssj.util.sort.*;

public class AsianOptionTest extends ArrayOfComparableChainsNN<AsianOptionComparable2> {

	public AsianOptionTest(AsianOptionComparable2 baseChain, String[] fileNames) {
		super(baseChain,fileNames);
		// this.baseChain = baseChain;
	}

	// Applies Array-RQMC for asian with various point sets and given sort, m
	// times independently.
	// sortCoord is the number of coordinates used to sort the points.

	// dimState is the dimension of the state that is used for the sort.
	// The points are sorted based on the first dimState coordinates.
	// numSets is the number of point sets for which the experiment is made.
	//
	/*
	 * public void testMethods (AsianOptionComparable2 asian,
	 * MultiDimSort<AsianOptionComparable2> sort, int sortCoordPts, int numSteps,
	 * int m, int numSets) {
	 */
	public void testMethods(AsianOptionComparable2 asian, MultiDimSort sort, int sortCoordPts, int numSteps, int m,
			int numSets) {
		int s; // Index of point set.
		/*
		 * int[] N = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
		 * 262144, 524288, 1048576 }; // logn from 8 to 20. int[] a = { 55, 115, 851,
		 * 1553, 2839, 6685, 9945, 12421, 38295, 114789, 177473, 286857, 271251 };
		 */
		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 }; // logn from 8
																											// to 20.
		int[] a = { 115, 851, 1553, 2839, 6685, 9945, 12421, 38295, 114789, 177473, 286857, 271251 };
		PointSet[] pointSets = new PointSet[numSets];
		PointSetRandomization randShift = new RandomShift(new MRG32k3a());
		PointSetRandomization randDigital = new LMScrambleShift(new MRG32k3a());
		PointSetRandomization randNUS = new NestedUniformScrambling(new MRG32k3a(), 31); // Scramble all 31 bits.
		// Monte Carlo experiments with nMC independent runs.
		int nMC = 1000 * 1000; // n to estimate MC variance.
		Tally statMC = new Tally();
		statMC.init();
		asian.simulRunsWithSubstreams(nMC, asian.d, new MRG32k3a(), statMC);
		double varMC = statMC.variance();
		System.out.println("\n\n --------------------------");
		System.out.println("MC average  = " + statMC.average());
		System.out.println("MC variance = " + varMC);

		/*
		 * * // Latin hypercube for (s = 0; s < numSets; ++s) { pointSets[s] = new
		 * LatinHypercube(N[s], stateDim+1); } // testVarianceRate (pointSets, rand,
		 * asian.d, m, varMC, "Asian-var-LHS", "Latin hypercube");
		 */

		/*
		 * // Independent points (Monte Carlo) for (s = 0; s < numSets; ++s) {
		 * pointSets[s] = new IndependentPointsCached(N[s],
		 * asian.getStateDimension()+1); } System.out.println (testVarianceRateFormat
		 * (pointSets, randShift, sort, 0, numSteps, m, varMC,
		 * "Asian-var-IndependentPoint", "Independent points"));
		 * 
		 * // Stratification int k; for (s = 0; s < numSets; ++s) { k = (int)
		 * Math.round(Math.pow(Num.TWOEXP[s + 9], 1.0 / (double) (sortCoordPts + 1)));
		 * pointSets[s] = new StratifiedUnitCube (k, sortCoordPts + 1); // Here the
		 * points must be sorted at each step, always. // In the case of Hilbert map,
		 * the points should be 2d and sorted based on one coordinate, // whereas the
		 * states are 2d and sorted by the Hilbert sort. } System.out.println
		 * (testVarianceRateFormat (pointSets, randShift, sort, sortCoordPts, numSteps,
		 * m, varMC, "Asian-var-Stratification", "Stratification"));
		 */

		// Sobol
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new SobolSequence(s + 9, 31, 1 + 1);
			else
				// The points are sorted here, but only once.
				pointSets[s] = new SortedAndCutPointSet(new SobolSequence(s + 9, 31, sortCoordPts + 1), sort);
		}
		// When the point set is sorted only one, it must be sorted before calling this
		// function.
		System.out.println(testVarianceRateFormat(pointSets, randDigital, sort, 0, numSteps, m, varMC,
				"Asian-var-sobol", "Sobol+LMS"));
		/*
		 * 
		 * // Sobol + Baker for (s = 0; s < numSets; ++s) { if (sortCoordPts == 1)
		 * pointSets[s] = new BakerTransformedPointSet (new SobolSequence(s + 9, 31,
		 * 1+1)); else pointSets[s] = new SortedAndCutPointSet (new
		 * BakerTransformedPointSet (new SobolSequence(s + 9, 31, sortCoordPts+1)),
		 * sort); }
		 * 
		 * System.out.println (testVarianceRateFormat (pointSets, randDigital, sort, 0,
		 * numSteps, m, varMC, "Asian-var-Sobol+LMS+Baker", "Sobol + LMS+ Baker"));
		 * 
		 * 
		 * // Sobol NUS for (s = 0; s < numSets; ++s) { if (sortCoordPts == 1) {
		 * CachedPointSet p = new CachedPointSet(new SobolSequence(s + 9, 31, 1+1));
		 * p.setRandomizeParent(false); pointSets[s] = p; } else { CachedPointSet p =
		 * new CachedPointSet(new SobolSequence(s + 9, 31, sortCoordPts+1));
		 * p.setRandomizeParent(false); // The points are sorted here, but only once.
		 * pointSets[s] = new SortedAndCutPointSet (p, sort); } } // When the point set
		 * is sorted only one, it must be sorted before calling this function.
		 * System.out.println (testVarianceRateFormat (pointSets, randNUS, sort, 0,
		 * numSteps, m, varMC, "Asian-var-Sobol+NUS", "Sobol + NUS\n")); //Korobov
		 * 
		 * for (s = 0; s < numSets; ++s){ if (sortCoordPts == 1) pointSets[s] = new
		 * KorobovLattice(N[s],a[s],1+1, 1); else pointSets[s] = new
		 * SortedAndCutPointSet (new KorobovLattice(N[s],a[s],sortCoordPts+1, 1),sort);
		 * }
		 * 
		 * System.out.println (testVarianceRateFormat (pointSets, randShift, sort, 0,
		 * numSteps, m, varMC, "Asian-var-korobov", "Korobov ")); // Korobov +baker for
		 * (s = 0; s < numSets; ++s) { if (sortCoordPts == 1) pointSets[s] = new
		 * BakerTransformedPointSet (new KorobovLattice (N[s], a[s], 1+1, 1)); else //
		 * The points are sorted here, but only once. pointSets[s] = new
		 * SortedAndCutPointSet (new BakerTransformedPointSet (new KorobovLattice (N[s],
		 * a[s], sortCoordPts+1, 1)), sort); } System.out.println
		 * (testVarianceRateFormat (pointSets, randShift, sort, 0, numSteps, m, varMC,
		 * "Asian-var-korobov-Baker", "Korobov + Baker"));
		 */
	}

	public void testMethods(AsianOptionComparable2 asian, int sortCoordPts, int numSteps, int m,
			int numSets) throws IOException {
		int s; // Index of point set.
		/*
		 * int[] N = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
		 * 262144, 524288, 1048576 }; // logn from 8 to 20. int[] a = { 55, 115, 851,
		 * 1553, 2839, 6685, 9945, 12421, 38295, 114789, 177473, 286857, 271251 };
		 */
		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 }; // logn from 8
																											// to 20.
		int[] a = { 115, 851, 1553, 2839, 6685, 9945, 12421, 38295, 114789, 177473, 286857, 271251 };
		PointSet[] pointSets = new PointSet[numSets];
		PointSetRandomization randShift = new RandomShift(new MRG32k3a());
		PointSetRandomization randDigital = new LMScrambleShift(new MRG32k3a());
		PointSetRandomization randNUS = new NestedUniformScrambling(new MRG32k3a(), 31); // Scramble all 31 bits.
		// Monte Carlo experiments with nMC independent runs.
		int nMC = 1000 * 1000; // n to estimate MC variance.
		Tally statMC = new Tally();
		statMC.init();
		asian.simulRunsWithSubstreams(nMC, asian.d, new MRG32k3a(), statMC);
		double varMC = statMC.variance();
		System.out.println("\n\n --------------------------");
		System.out.println("MC average  = " + statMC.average());
		System.out.println("MC variance = " + varMC);

		/*
		 * * // Latin hypercube for (s = 0; s < numSets; ++s) { pointSets[s] = new
		 * LatinHypercube(N[s], stateDim+1); } // testVarianceRate (pointSets, rand,
		 * asian.d, m, varMC, "Asian-var-LHS", "Latin hypercube");
		 */

		/*
		 * // Independent points (Monte Carlo) for (s = 0; s < numSets; ++s) {
		 * pointSets[s] = new IndependentPointsCached(N[s],
		 * asian.getStateDimension()+1); } System.out.println (testVarianceRateFormat
		 * (pointSets, randShift, sort, 0, numSteps, m, varMC,
		 * "Asian-var-IndependentPoint", "Independent points"));
		 * 
		 * // Stratification int k; for (s = 0; s < numSets; ++s) { k = (int)
		 * Math.round(Math.pow(Num.TWOEXP[s + 9], 1.0 / (double) (sortCoordPts + 1)));
		 * pointSets[s] = new StratifiedUnitCube (k, sortCoordPts + 1); // Here the
		 * points must be sorted at each step, always. // In the case of Hilbert map,
		 * the points should be 2d and sorted based on one coordinate, // whereas the
		 * states are 2d and sorted by the Hilbert sort. } System.out.println
		 * (testVarianceRateFormat (pointSets, randShift, sort, sortCoordPts, numSteps,
		 * m, varMC, "Asian-var-Stratification", "Stratification"));
		 */

		// Sobol
		for (s = 0; s < numSets; ++s) {
			
				pointSets[s] = new SobolSequence(s + 9, 31, 1 + 1);
		
		}
		// When the point set is sorted only one, it must be sorted before calling this
		// function.
		System.out.println(testVarianceRateFormat(pointSets, randDigital, 0, numSteps, m, varMC,
				"Asian-var-sobol", "Sobol+LMS"));
		/*
		 * 
		 * // Sobol + Baker for (s = 0; s < numSets; ++s) { if (sortCoordPts == 1)
		 * pointSets[s] = new BakerTransformedPointSet (new SobolSequence(s + 9, 31,
		 * 1+1)); else pointSets[s] = new SortedAndCutPointSet (new
		 * BakerTransformedPointSet (new SobolSequence(s + 9, 31, sortCoordPts+1)),
		 * sort); }
		 * 
		 * System.out.println (testVarianceRateFormat (pointSets, randDigital, sort, 0,
		 * numSteps, m, varMC, "Asian-var-Sobol+LMS+Baker", "Sobol + LMS+ Baker"));
		 * 
		 * 
		 * // Sobol NUS for (s = 0; s < numSets; ++s) { if (sortCoordPts == 1) {
		 * CachedPointSet p = new CachedPointSet(new SobolSequence(s + 9, 31, 1+1));
		 * p.setRandomizeParent(false); pointSets[s] = p; } else { CachedPointSet p =
		 * new CachedPointSet(new SobolSequence(s + 9, 31, sortCoordPts+1));
		 * p.setRandomizeParent(false); // The points are sorted here, but only once.
		 * pointSets[s] = new SortedAndCutPointSet (p, sort); } } // When the point set
		 * is sorted only one, it must be sorted before calling this function.
		 * System.out.println (testVarianceRateFormat (pointSets, randNUS, sort, 0,
		 * numSteps, m, varMC, "Asian-var-Sobol+NUS", "Sobol + NUS\n")); //Korobov
		 * 
		 * for (s = 0; s < numSets; ++s){ if (sortCoordPts == 1) pointSets[s] = new
		 * KorobovLattice(N[s],a[s],1+1, 1); else pointSets[s] = new
		 * SortedAndCutPointSet (new KorobovLattice(N[s],a[s],sortCoordPts+1, 1),sort);
		 * }
		 * 
		 * System.out.println (testVarianceRateFormat (pointSets, randShift, sort, 0,
		 * numSteps, m, varMC, "Asian-var-korobov", "Korobov ")); // Korobov +baker for
		 * (s = 0; s < numSets; ++s) { if (sortCoordPts == 1) pointSets[s] = new
		 * BakerTransformedPointSet (new KorobovLattice (N[s], a[s], 1+1, 1)); else //
		 * The points are sorted here, but only once. pointSets[s] = new
		 * SortedAndCutPointSet (new BakerTransformedPointSet (new KorobovLattice (N[s],
		 * a[s], sortCoordPts+1, 1)), sort); } System.out.println
		 * (testVarianceRateFormat (pointSets, randShift, sort, 0, numSteps, m, varMC,
		 * "Asian-var-korobov-Baker", "Korobov + Baker"));
		 */
	}
	
	public static void main(String[] args) throws IOException {
		double r = Math.log(1.09);
		// int d = 12;
		int d = 4;
		// double t1 = (240.0 - d + 1) / 365.0;
		// double T = 240.0 / 365.0;
		double t1 = 1.0 / d;
		double T = 1.0;
		double K = 100.0;
		double s0 = 100.0;
		double sigma = 0.5;
		// double sigma = 0.2;
		// int numSteps = d;

		int m = 50; // Number of replications.
		 int numSets = 9; // Number of point sets to try.
//		int numSets = 8; // Number of point sets to try.
		// int numSets = 2; // Number of point sets to try.

		AsianOptionComparable2 asian = new AsianOptionComparable2(r, d, t1, T, K, s0, sigma);
		String [] fileNames = new String[d];
		String base = "./data/asian/";
		for(int i = 0; i < d; i++) {
			fileNames[i] = base + "Asian_Step" + i + ".zip";
		}
		AsianOptionTest test = new AsianOptionTest(asian,fileNames); // This is the array of comparable chains.
		// PointSetRandomization rand = new RandomShift(new MRG32k3a());

		MultiDimSort<AsianOptionComparable2> sort;
		MultiDimSort01 sort01;
		// MultiDimSortN<MultiDim> sort02;
		MultiDimSortN sort02;
		double[] batchExp = { 0.5, 0.5 };

		System.out.println(asian.toString());

		
//		  System.out.println("\n *************  SPLIT SORT  *************** \n"); sort
//		  = new SplitSort (2); // test = new AsianOptionTest (asian, rand, sort);
//		  test.testMethods (asian, sort, 2, d,m, numSets);
		  
//		  System.out.println("\n *************  BATCH SORT  *************** \n"); sort
//		  = new BatchSort<AsianOptionComparable2>(batchExp); // Sort in 2 dim.
//		  test.testMethods(asian, sort, 2, d, m, numSets);
		  
//		  System.out.println("\n **********  HILBERT BATCH SORT  ***********\n"); sort
//		  = new HilbertCurveBatchSort<AsianOptionComparable2>(batchExp, 20);
//		  test.testMethods(asian, sort, 1, d, m, numSets);
		 
//		System.out.println("\n *************  HILBERT SORT  *************** \n");
//		sort01 = new HilbertCurveSort(2, 12);
//		test.testMethods(asian, sort01, 1, d, m, numSets);

		System.out.println("\n *************  NeuralNetwork  SORT  *************** \n");
//		sort02 = new NeuralNetworkSort(2);
		// test.testMethods (biology, sort02, 1, T, m, numSets, "NeuralNetworkSort");
		// sort02 = new NeuralNetworkSort(2);
		test.testMethods(asian, 1, d, m, numSets);

		System.out.println("\n  Done !!!");
	}
}
