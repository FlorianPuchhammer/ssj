package flo.optionsArrayRQMC;

import umontreal.ssj.markovchainrqmc.*;

import java.io.FileWriter;
import java.io.IOException;

import umontreal.ssj.charts.HistogramChart;
import umontreal.ssj.hups.*;
import umontreal.ssj.rng.*;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.TallyStore;
import umontreal.ssj.util.*;
import umontreal.ssj.util.sort.BatchSort;
import umontreal.ssj.util.sort.HilbertCurveBatchSort;
import umontreal.ssj.util.sort.HilbertCurveSort;
import umontreal.ssj.util.sort.MultiDim01;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.MultiDimSort01;
import umontreal.ssj.util.sort.SplitSort;

public class VGAsianOptionTest extends ArrayOfComparableChains<VGAsianOptionComparable> {

	public VGAsianOptionTest(VGAsianOptionComparable baseChain) {
		super(baseChain);
		// this.baseChain = baseChain;
	}

	// Applies Array-RQMC for asian with various point sets and given sort, m
	// times independently.
	// sortCoord is the number of coordinates used to sort the points.

	// dimState is the dimension of the state that is used for the sort.
	// The points are sorted based on the first dimState coordinates.
	// numSets is the number of point sets for which the experiment is made.
	//
	public void testMethods(VGAsianOptionComparable asian, MultiDimSort sort, int sortCoordPts, int numSteps, int m,
			int numSets, String sor) throws IOException {
		int s; // Index of point set.
		/*
		 * int[] N = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
		 * 262144, 524288, 1048576 }; // logn from 8 to 20.
		 */
		/*
		 * int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
		 * 524288, 1048576 }; // logn from 9 to 20.
		 */ /*
			 * int[] a = { 115, 851, 1553, 2839, 6685, 9945, 12421, 38295, 114789, 177473,
			 * 286857, 271251 }; // logn from 9 to 20.
			 */
		/*
		 * int[] N = { 65536, 131072, 262144, 524288, 1048576 }; // logn from 16 to 20.
		 * int[] a = { 38295, 114789, 177473, 286857, 271251 }; // logn from 16 to 20.
		 * int mink = 16;
		 */
		
		/*int[] N = {  512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
				131072, 262144, 524288, 1048576 }; // logn from 8 to 20.
		int[] a = {  115, 851, 1553, 2839, 6685, 9945, 12421, 38295, 114789,
				177473, 286857, 271251 };
		int mink =  9; */
		
		int[] N = { 65536,
				131072, 262144, 524288, 1048576 }; // logn from 16 to 20.
		//int[] a = {  38295, 114789, 177473, 286857, 271251 };
		int [] [] a= {

				 {1, 19463, 8279, 14631},

				 {1, 50687, 44805, 12937},

				 {1, 100135, 28235, 39865},

				 {1, 154805, 242105, 171449},

				 {1, 387275, 314993, 50301}

				 };
		int mink = 16;
		PointSet[] pointSets = new PointSet[numSets];
		PointSetRandomization randShift = new RandomShift(new MRG32k3a());
		PointSetRandomization randDigital = new LMScrambleShift(new MRG32k3a());
		PointSetRandomization randNUS = new NestedUniformScrambling(new MRG32k3a());

		// Monte Carlo experiments with nMC independent runs.
		int nMC =  1000 * 1000; // n to estimate MC variance.
		TallyStore statMC = new TallyStore();
		statMC.init();
		HistogramChart chartH;
		asian.simulRunsWithSubstreams(nMC, asian.d, new MRG32k3a(), statMC);
		double varMC = statMC.variance();
		System.out.println("\n\n--------------------------");
		System.out.println("MC average  = " + statMC.average());
		System.out.println("MC variance = " + varMC);
		// Independent points (Monte Carlo) //
		String outFile = "VGFlattice" + sor +".txt";
		StringBuffer sb = new StringBuffer("");
		String str;
		for (s = 0; s < numSets; ++s) {
			pointSets[s] = new IndependentPointsCached(N[s], 4);
		}
		str = testVarianceRateFormat(pointSets, randShift, sort, sortCoordPts, numSteps, m, varMC,
				sor + "VGF-var-indep", "Independent points");
		System.out.println(str);
		sb.append(str + "\n");

//		  Latin hypercube for (s = 0; s < numSets; ++s) { pointSets[s] = new
//		  LatinHypercube(N[s], stateDim+1); } // testVarianceRate (pointSets,
//		  rand, asian.d, m, varMC, "Asian-var-LHS", "Latin hypercube");

		// Stratification
		int k;
		for (s = 0; s < numSets; ++s) {
			k = (int) Math.round(Math.pow(Num.TWOEXP[s + mink], 1.0 / (double) (sortCoordPts + 2)));

			pointSets[s] = new StratifiedUnitCube(k, sortCoordPts + 2);
			// Here the points must be sorted at each step, always.
			// In the case of Hilbert map, the points should be 2d and sorted based on one
			// coordinate,
			// whereas the states are 2d and sorted by the Hilbert sort.
		}
		str =testVarianceRateFormat(pointSets, randShift, sort, sortCoordPts, numSteps, m, varMC,
				sor + "VGF-var-Stratification", " Stratification");
		System.out.println(str);
		sb.append(str + "\n");

		// Sobol + LMS
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new SobolSequence((int) Math.pow(2, s + mink), 3);
			else
				// The points are sorted here, but only once.
				pointSets[s] = new SortedAndCutPointSet(
						new SobolSequence((int) Math.pow(2, s + mink), sortCoordPts + 2), sort);
		}
		// When the point set is sorted only one, it must be sorted before calling this
		// function.
		str =testVarianceRateFormat(pointSets, randDigital, sort, 0, numSteps, m, varMC,
				sor + "VGF-var-Sobol + LMS", "Sobol + LMS");
		System.out.println(str);
		sb.append(str + "\n");

		// Sobol + LMS + Baker
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new BakerTransformedPointSet(new SobolSequence(s + mink, 31, 3));
			else
				pointSets[s] = new BakerTransformedPointSet(new SobolSequence(s + mink, 31, sortCoordPts + 2));

		}
		str =testVarianceRateFormat(pointSets, randDigital, sort, 0, numSteps, m, varMC,
				sor + "VGF-var-Sobol + LMS + Baker", "Sobol + LMS+Baker");
		System.out.println(str);
		sb.append(str + "\n");
		// Sobol NUS
				for (s = 0; s < numSets; ++s) {
					if (sortCoordPts == 1) {
						CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31, 3));
						p.setRandomizeParent(false);
						pointSets[s] = p;
					}
					else {
						CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31, sortCoordPts + 2));
						p.setRandomizeParent(false);
						// The points are sorted here, but only once.
					    pointSets[s] = new SortedAndCutPointSet (p, sort);
					 }
				}
				// When the point set is sorted only one, it must be sorted before calling this function.
				str =testVarianceRateFormat (pointSets, randNUS, sort, 0,
						 numSteps, m, varMC, sor + "VGF-var-Sobol + NUS", "Sobol + NUS");
				System.out.println (str);
				sb.append(str + "\n");
/*
		// Korobov
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new BakerTransformedPointSet(new KorobovLattice(N[s], a[s], 1 + 2, 1));
			else
				// The points are sorted here, but only once.
				pointSets[s] = new SortedAndCutPointSet(
						new BakerTransformedPointSet(new KorobovLattice(N[s], a[s], sortCoordPts + 2, 1)), sort);
		}
		str =testVarianceRateFormat(pointSets, randShift, sort, 0, numSteps, m, varMC,
				sor+"VGF-var-korobov", " Korobov");
		System.out.println(str);
		sb.append(str + "\n");
		// Korobov + baker

		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new BakerTransformedPointSet(new KorobovLattice(N[s], a[s], 1+2, 1));
			else
				// The points are sorted here, but only once.
				pointSets[s] = new SortedAndCutPointSet(
						new BakerTransformedPointSet(new KorobovLattice(N[s], a[s], sortCoordPts + 2, 1)), sort);
		}
		str =testVarianceRateFormat(pointSets, randShift, sort, 0, numSteps, m, varMC,
				sor+"VGF-var-korobov-Baker", " Korobov + Baker");
		System.out.println(str);
		sb.append(str + "\n");
		// Korobov
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new BakerTransformedPointSet(new KorobovLattice(N[s], a[s], 1 + 2, 0));
			else
				// The points are sorted here, but only once.
				pointSets[s] = new SortedAndCutPointSet(
						new BakerTransformedPointSet(new KorobovLattice(N[s], a[s], sortCoordPts + 2, 0)), sort);
		}
		str =testVarianceRateFormat(pointSets, randShift, sort, 0, numSteps, m, varMC,
				sor+"VGF-var-korobovstandard", " Korobovstandard");
		System.out.println(str);
		sb.append(str + "\n");
		// Korobov + baker

		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new BakerTransformedPointSet(new KorobovLattice(N[s], a[s], 1+2, 0));
			else
				// The points are sorted here, but only once.
				pointSets[s] = new SortedAndCutPointSet(
						new BakerTransformedPointSet(new KorobovLattice(N[s], a[s], sortCoordPts + 2, 0)), sort);
		}
		str =testVarianceRateFormat(pointSets, randShift, sort, 0, numSteps, m, varMC,
				sor+"VGF-var-korobovstandard-Baker", " Korobovstandard + Baker");
		System.out.println(str);
		sb.append(str + "\n");*/

		// lattice
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new Rank1Lattice(N[s], a[s], 1 + 2);
			else
				pointSets[s] = new SortedAndCutPointSet(new Rank1Lattice(N[s], a[s], sortCoordPts + 2), sort);
		}
		str =testVarianceRateFormat(pointSets, randShift, sort, 0, numSteps, m, varMC,
				sor + "VGF-var-Lattice", "Lattice");
		System.out.println(str);
		sb.append(str + "\n");

		// lattice + baker
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new BakerTransformedPointSet(new Rank1Lattice(N[s], a[s], 1 + 2));
			else
				// The points are sorted here, but only once.
				pointSets[s] = new SortedAndCutPointSet(
						new BakerTransformedPointSet(new Rank1Lattice(N[s], a[s], sortCoordPts + 2)), sort);
		}
		str =testVarianceRateFormat(pointSets, randShift, sort, 0, numSteps, m, varMC,
				sor + "VGF-var-Lattice+baker", "Lattice+baker");
		System.out.println(str);
		sb.append(str + "\n");
		
		FileWriter file = new FileWriter(outFile);
		file.write(sb.toString());
		file.close();
	}

	/*
	 * statMC.setConfidenceIntervalStudent(); chartH = new HistogramChart("",
	 * "Payoff", "Frequency", statMC); chartH.view(800,500);
	 */

	public static void usage() {
		System.err.println(
				"usage: VGTest { split | batch | hilbertbatch | hilbert { normal | logistic <x0> <w> | logisticGC <mu> <c> } }");
		System.err.println("examples:");
		System.err.println("    VGTest split");
		System.err.println("    VGTest batch");
		System.err.println("    VGTest hilbert normal");
		System.err.println("    VGTest hilbert logistic 0.0 2.0");
	}

	public static void main(String[] args) throws IOException {

//		int d = 10;
//		double t1 = (240.0 - d + 1) / 365.0;
//		//double t1 = 231 / 365.0;
//		
//		double T = 240.0 / 365.0;
//		double K = 100.0;
//		double s0 = 100.0;
//		//double r=0.0548;
//		//double alpha = 1.0;
//		//double beta = 1e-6;//1/Math.pow(10, 6);
//		//double sigma=0.1927;
//		//double teta = -0.2859 ;
//	    double r = Math.log(1.09);
//		double beta = 1e-9;
//	    //double beta = 0;
//		double teta = 0 ;
//		double sigma = 0.2 ;
//		double alpha = 1;

		int d = 10;
		double t1 = (240.0 - d + 1) / 365.0;
		// double t1 = 231 / 365.0;

		double T = 240.0 / 365.0;
		double K = 100.0;
		double s0 = 100.0;
		// double r=0.0548;
		// double alpha = 1.0;
		// double beta = 1e-6;//1/Math.pow(10, 6);
		// double sigma=0.1927;
		// double teta = -0.2859 ;
		double r = 0.1;
		double nu = 0.3;
		// double beta = 0;
		double teta = -0.1436;
		double sigma = 0.12136;
		double muu = 1;

		// int numSteps = d;

		//int m = 100; // Number of replications.
		//int numSets = 12; // Number of point sets to try.
int m=100;
int numSets = 5;
		//VGAsianOptionComparable.RealsTo01Map map = null;

		// Parse command-line options.

		if (args.length < 1) {
			usage();
			System.exit(1);
		}

		/*
		 * VGAsianOptionComparable asian = new VGAsianOptionComparable(r, d, t1, T, K,
		 * beta, alpha, teta, s0, sigma, map); VGAsianOptionTest3 test = new
		 * VGAsianOptionTest3(asian); // This is the array of comparable chains. //
		 * PointSetRandomization rand = new RandomShift(new MRG32k3a());
		 * MultiDimSort<VGAsianOptionComparable> sort; MultiDimSort01<MultiDim01>
		 * sort01; String sortType = args[0];
		 */

		/*
		 * if (sortType.equals("hilbert")) {
		 * 
		 * if (args.length == 2 && args[1].equals("normal")) {
		 * System.out.println("Using normal CDF map"); map = new
		 * VGAsianOptionComparable.NormalCDFMap();
		 * 
		 * } else if (args.length == 4 && args[1].equals("logisticGC")) { double mu =
		 * Double.parseDouble(args[2]); double c = Double.parseDouble(args[3]);
		 * System.out.println("Using Gerber & Chopin logistic map centered at " + mu +
		 * " with half width " + c); map = new VGAsianOptionComparable.LogisticGCMap(mu
		 * - c, mu + c); } else if (args.length == 4 && args[1].equals("logistic")) {
		 * double x0 = Double.parseDouble(args[2]); double w =
		 * Double.parseDouble(args[3]);
		 * System.out.println("Using logistic map centered at " + x0 + " with scale " +
		 * w); map = new VGAsianOptionComparable.LogisticMap(x0, w); } else { usage();
		 * System.exit(1); }
		 * 
		 * }
		 * 
		 * else { if (sortType.equals("others")) {
		 * 
		 * double[] batchExp = { 0.5, 0.5};
		 * 
		 * 
		 * System.out.println("\n******************************************\n");
		 * System.out.println("\nsort type: SPLIT SORT\n"); sort = new
		 * SplitSort<VGAsianOptionComparable> (2); test.testMethods (asian, sort, 2, d,
		 * m, numSets, "split");
		 * 
		 * 
		 * System.out.println("\n******************************************");
		 * System.out.println("\nsort type: BATCH SORT\n"); sort = new
		 * BatchSort<VGAsianOptionComparable>(batchExp); test.testMethods(asian, sort,
		 * 2, d, m, numSets,"batch");
		 * 
		 * 
		 * System.out.println("\n******************************************");
		 * System.out.println("\nsort type: HILBERT BATCH SORT\n"); sort = new
		 * HilbertCurveBatchSort<VGAsianOptionComparable>(batchExp, 20);
		 * test.testMethods(asian, sort, 1, d, m, numSets,"hilbertbatch");
		 * 
		 * 
		 * 
		 * System.out.println("\n******************************************");
		 * System.out.println("\nsort type: HILBERT SORT\n"); sort01 = new
		 * HilbertCurveSort(2, 12); test.testMethods(asian, sort01, 1, d, m, numSets,
		 * "hilbert");
		 * 
		 * 
		 * } } if (sortType.equals("hilbert")) {
		 * System.out.println("\n******************************************");
		 * System.out.println("\nsort type: HILBERT SORT\n"); sort01 = new
		 * HilbertCurveSort(2, 12); test.testMethods(asian, sort01, 1, d, m, numSets,
		 * "hilbert");}
		 * 
		 * 
		 * // double[] batchExp = { 0.3, 0.3, 0.4}; double[] batchExp = { 0.5, 0.5 };
		 * 
		 * if (sortType.equals("split")) {
		 * System.out.println("\n******************************************\n");
		 * System.out.println("\nsort type: SPLIT SORT\n"); sort = new
		 * SplitSort<VGAsianOptionComparable>(2); test.testMethods(asian, sort, 2, d, m,
		 * numSets, "split"); } else if (sortType.equals("batch")) {
		 * System.out.println("\n******************************************");
		 * System.out.println("\nsort type: BATCH SORT\n"); sort = new
		 * BatchSort<VGAsianOptionComparable>(batchExp); // Sort in 3 dim.
		 * test.testMethods(asian, sort, 2, d, m, numSets, "batch"); } else if
		 * (sortType.equals("hilbertbatch")) {
		 * System.out.println("\n******************************************");
		 * System.out.println("\nsort type: HILBERT BATCH SORT\n"); sort = new
		 * HilbertCurveBatchSort<VGAsianOptionComparable>(batchExp, 20);
		 * test.testMethods(asian, sort, 1, d, m, numSets, "hilbertbatch"); } else if
		 * (sortType.equals("hilbert")) {
		 * System.out.println("\n******************************************");
		 * System.out.println("\nsort type: HILBERT SORT\n"); sort01 = new
		 * HilbertCurveSort(2, 12); test.testMethods(asian, sort01, 1, d, m, numSets,
		 * "hilbert"); } else { System.out.println("invalid sort type!"); usage();
		 * System.exit(1); }
		 */
		
		VGAsianOptionComparable.RealsTo01Map map = null;


		// Parse command-line options.

		if (args.length < 1) {
			usage();
			System.exit(1);
		}

		String sortType = args[0];

		if (sortType.equals("hilbert")) {

			if (args.length == 2 && args[1].equals("normal")) {
				System.out.println("Using normal CDF map");
				map = new VGAsianOptionComparable.NormalCDFMap();
			}
			else if (args.length == 4 && args[1].equals("logisticGC")) {
				double mu = Double.parseDouble(args[2]);
				double c = Double.parseDouble(args[3]);
				System.out.println("Using Gerber & Chopin logistic map centered at " + mu + " with half width " + c);
				map = new VGAsianOptionComparable.LogisticGCMap(mu - c, mu + c);
			}
			else if (args.length == 4 && args[1].equals("logistic")) {
				double x0 = Double.parseDouble(args[2]);
				double w = Double.parseDouble(args[3]);
				System.out.println("Using logistic map centered at " + x0 + " with scale " + w);
				map = new VGAsianOptionComparable.LogisticMap(x0, w);
			}
			else {
				usage();
				System.exit(1);
			}
		}


		VGAsianOptionComparable asian = new VGAsianOptionComparable (r, d, t1, T, K,nu,muu,teta,s0, sigma, map);
		VGAsianOptionTest test = new VGAsianOptionTest (asian);    // This is the array of comparable chains.
		// PointSetRandomization rand = new RandomShift(new MRG32k3a());
		MultiDimSort<VGAsianOptionComparable> sort;
		MultiDimSort01 sort01;

		//double[] batchExp = { 0.3, 0.3, 0.4};
		double[] batchExp = { 0.25, 0.75};

		if (sortType.equals("split")) {
			System.out.println("\n******************************************\n");
			System.out.println("\nsort type: SPLIT SORT\n");
			sort = new SplitSort<VGAsianOptionComparable> (2);
			test.testMethods (asian, sort, 2, d, m, numSets,"split4");
		}
		else if (sortType.equals("batch")) {
			System.out.println("\n******************************************");
			System.out.println("\nsort type: BATCH SORT\n");
			sort = new BatchSort<VGAsianOptionComparable>(batchExp);    // Sort in 3 dim.
			test.testMethods(asian, sort, 2, d, m, numSets,"batch4");
		}
		else if (sortType.equals("hilbertbatch")) {
			System.out.println("\n******************************************");
			System.out.println("\nsort type: HILBERT BATCH SORT\n");
			sort = new HilbertCurveBatchSort<VGAsianOptionComparable>(batchExp, 20);
			test.testMethods(asian, sort, 1, d, m, numSets,"hilbertbatch4");
		}
		else if (sortType.equals("hilbert")) {
			System.out.println("\n******************************************");
			System.out.println("\nsort type: HILBERT SORT\n");
			sort01 = new HilbertCurveSort(2, 12);
			test.testMethods(asian, sort01, 1, d, m, numSets,"hilbertlattice9"+args [1]);
		}
		else {
			System.out.println("invalid sort type!");
			usage();
			System.exit(1);
		}


		System.out.println("\n  Done !!!");
	}
}
