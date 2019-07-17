package flo.biologyArrayRQMC.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

import umontreal.ssj.functionfit.LeastSquares;
import umontreal.ssj.hups.BakerTransformedPointSet;
import umontreal.ssj.hups.CachedPointSet;
import umontreal.ssj.hups.IndependentPointsCached;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.NestedUniformScrambling;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.RandomShift;
import umontreal.ssj.hups.Rank1Lattice;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.hups.SortedAndCutPointSet;
import umontreal.ssj.hups.StratifiedUnitCube;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChains;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChainsMultipleSorts;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChainsNN;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChainsSubsteps;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.markovchainrqmc.MarkovChainComparableWithSubsteps;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.Num;
import umontreal.ssj.util.sort.BatchSort;
import umontreal.ssj.util.sort.HilbertCurveBatchSort;
import umontreal.ssj.util.sort.HilbertCurveSort;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.SplitSort;

public class testChemicalReactionNetworkSubsteps {

	public static void main(String[] args) throws IOException {

		ChemicalReactionNetworkSubsteps model;

		/*
		 * ******************* LINEAR BIRTH DEATH
		 **********************/
		double[] c = { 0.1, 0.1 };
		double[] x0 = { 1000.0 };
		double T = 1.6;
		double tau = T / 8.0;
		model = new LinearBirthDeathSubsteps(c, x0, tau, T);

		/*
		 * ******************* PKA
		 **********************/
//		double[]c = {8.696E-5, 0.02, 1.154E-4,0.02,0.016,0.0017};//Nano: 1E-9
//		double[] x0 = {33000.0,33030.0, 1100.0, 1100.0, 1100.0, 1100.0};
//		double T = 0.00005;
//		double tau = T/20.0;

//		 model = new PKA(c,x0,tau,T);

		/*
		 * ******************* SCHLOEGL PROJECTED
		 **********************/
//		double[] c = { 3E-7, 1E-4, 1E-3, 3.5 };
//		double[] x0 = { 250.0, 1E5};
//		double N0 = 2E5 + 1E5 + 250.0;
//		double T = 4.0;
//		double tau = T/20.0;
//
//		model = new SchloeglSystemProjected(c, x0, tau, T,N0);

		/*
		 * ******************* REVERSIBLE ISO PROJECTED
		 **********************/
//		double epsInv = 1E2;
//		double alpha = 1E-4;
//		double[]c = {1.0,alpha};
//		double[] x0 = {epsInv};
//		double N0 = epsInv + epsInv/alpha;
//		double T = 1.6;
//		double tau = 0.2;
//
//		
//		
//		 model = new ReversibleIsomerizationProjected(c,x0,tau,T,N0);

		/*
		 * ******************* MAPK
		 **********************/
//		double[]c = {0.027,1.35,1.5,Math.log(2.0)*10.0,0.028,1.73,15.0,0.027,1.35,1.5,Math.log(2.0)*10.0,0.028,1.73,15.0};//Nano: 1E-9
//		double[] x0 = {2500.0,4500.0,8000.0,4500.0,7000.0,4500.0,2500.0,4500.0,4500.0,5500.0,7000.0};
//		double T = 0.01;
//		double tau = T/10.0;
//		
//		model = new MAPK(c,x0,tau,T);

		/*
		 * ******************* ENZYME-KINETICS
		 **********************/
//		double[] c = { 1E-3, 1E-4, 0.1 };// Nano: 1E-9
//		double[] x0 = { 200, 500, 200, 0 };
//		double T = 10.0;
//		double tau = T / 7.0;
//
//		model = new EnzymeKineticsSubsteps(c, x0, tau, T);

		/*
		 * ******************* ENZYME-KINETICS NO P
		 **********************/
//		double[]c = {1E-3,1E-4, 0.1};//Nano: 1E-9
//		double[] x0 = {200,500,200};
//		double T = 10.0;
//		double tau = T/7.0;
//		 model = new EnzymeKineticsNoP(c,x0,tau,T);
//		
		System.out.println(model.toString());

//		 String modelDescription = "RevIso";
//		 String modelDescription = "MAPK-single-" + "0-2-3-5-8-9-10-01-14-47-67" + "noBias" ;// + "_MORENOISE_18";
//		 String modelDescription = "Schloegl-single-0-1-00-01-000-001-nobias-LATTICE";
//		 String modelDescription = "Schloegl-single-0-1-00-01-000-nobias_MORENOISE_18";
		String modelDescription = "enzymeKineticsP-single-2-3-01" + "noBias" + "_MORENOISE_20";
//		 String modelDescription = "cAMP-single-1-2-3-01-12-45-011-112-455" + "noBias";
		boolean bias = false;

//		String dataFolder = "data/PKA/";
		model.init();

//		 int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
//		 524288, 1048576 }; // n from 8
		// to 20.
		int[] N = { 65536, 131072, 262144, 524288, 1048576 }; // n from 8
//		int[] N = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}; // n from 8

		int[] logN = { 16, 17, 18, 19, 20 };
		int mink = 16;
		int numSets = N.length;
//		numSets = 3;
		int m = 100; m=5;
//		m=10;
//		int[] a = {1, 387275, 314993, 465187, 61873};
//		int[] a =  {387275, 457903, 282967, 216675, 236269};
//		int[] a = {1, 444567, 243791, 106519, 489649, 501713}; //schloegl, sum
//		int[] a = {1, 387275, 314993, 465187, 61873, 284435}; //schloegl, max
//		int[][] a = {
//
//				{ 1, 19463, 8279 },
//
//				{ 1, 50687, 44805 },
//
//				{ 1, 100135, 28235 },
//
//				{ 1, 154805, 242105 },
//
//				{ 1, 387275, 314993 }
//
//		};
		int[][] a = {

				{ 1, 19463, 8279, 14631, 12629, 3257, 25367 },

				{ 1, 50687, 44805, 12937, 21433, 42925, 47259 },

				{ 1, 100135, 28235, 39865, 43103, 121135, 93235 },

				{ 1, 154805, 242105, 171449, 27859, 174391, 129075 },

				{ 1, 387275, 314993, 50301, 174023, 354905, 481763 }

		};

		int numSteps = (int) (T / tau);

		System.out.println("TEST0: numSteps = " + numSteps);
		System.out.println("TEST1: numSteps = " + Math.ceil(T / tau));

		int sortCoordPts;
		ArrayList<MultiDimSort> sortList = new ArrayList<MultiDimSort>();

		int rows = 262144; // rows = 1048576;
//		int cols = 7; //PKA
//		int cols = 12; //MAPK
		int cols = 5; // EnzymeKinetics
//		int cols = 3; //Schloegl

		Scanner sc;
		double[][] vars = new double[rows][cols - 1];
		double[] response = new double[rows];
		double[] reg;

//		sc = new Scanner(new BufferedReader(new FileReader("/u/puchhamf/misc/jars/biology/EnzymeKineticsP/data/"
//				+ "MCDataLessNoise_Step_" + (numSteps - 1) + ".csv")));
//		sc = new Scanner(new BufferedReader(
//				new FileReader("/u/puchhamf/misc/jars/biology/EnzymeKineticsP/data/" + "MCData_Step_" + (numSteps - 1) + ".csv")));
		sc = new Scanner( new BufferedReader(new FileReader("/u/puchhamf/misc/jars/biology/EnzymeKineticsP/data/" + "MCData_Step_" + 5 + ".csv")));

//		sc = new Scanner( new BufferedReader(new FileReader("data/Schloegl1/MCDataLessNoise_Step_" + 19 + ".csv")));

//		int[][] reducedCols = {{1},{2},{3},{0,1},{1,2},{4,5},{0,1,1},{1,1,2},{4,5,5}};   //PKA, linear
//		int[][] reducedCols = {{5}};   //PKA4
//		int[][] reducedCols = {{2},{3},{0,1},{0,2},{1,2},{2,2},{0,0,1},{0,1,1},{0,1,2},{0,0,1,1}}; //EnzymeKinetics
		int[][] reducedCols = { { 2 }, { 3 }, { 0, 1 } };
//		int[][] reducedCols = {{0}, {1}, {2}, {3}, {0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}, {1, 
//			  2}, {1, 3}, {2, 2}, {2, 3}, {3, 3}};
//		int[][] reducedCols = {{1,1,3},{1,2,2},{1,2,3},{1,3,3},{2,2,2},{2,2,3},{2,3,3},{3,3,3}};

//		int[][] reducedCols = {{0},{1}}; //Schloegl, linear
//		int[][] reducedCols = {{0},{1},{0,0},{0,1},{0,0,0},{0,0,1}};
//		int[][] reducedCols = {{0}, {1}, {2}, {3}, {0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}, {1, 
//			  2}, {1, 3}, {2, 2}, {2, 3}, {3, 3}, {0, 0, 0}, {0, 0, 1}, {0, 0, 
//				  2}, {0, 0, 3}, {0, 1, 1}, {0, 1, 2}, {0, 1, 3}, {0, 2, 2}, {0, 2, 
//				  3}, {0, 3, 3}, {1, 1, 1}, {1, 1, 2}, {1, 1, 3}, {1, 2, 2}, {1, 2, 
//				  3}, {1, 3, 3}, {2, 2, 2}, {2, 2, 3}, {2, 3, 3}, {3, 3, 3}};
//		int[][] reducedCols = {{0}, {1}, {2}, {3}, {0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}, {1, 
//			  2}, {1, 3}, {2, 2}, {2, 3}, {3, 3}};
//		int[][] reducedCols = {{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}}; //MAPK, linear
//		int[][] reducedCols = {{0},{2},{3},{5},{8},{9},{10},{0,1},{1,4},{4,7},{6,7}}; //MAPK6, linear

		double[][] varsReduced = new double[rows][reducedCols.length];

		for (int i = 0; i < rows; i++) {
			String[] line = sc.nextLine().trim().split(",");
			response[i] = Double.parseDouble(line[cols - 1]);
			for (int j = 0; j < cols - 1; j++) {
				vars[i][j] = Double.parseDouble(line[j]);
			}
			int j = 0;
			for (int[] tuples : reducedCols) {
				varsReduced[i][j] = 1.0;
				for (int col : tuples) {
					varsReduced[i][j] *= vars[i][col];
				}
				++j;
			}
		}
		sc.close();

//		 reg = LeastSquares.calcCoefficients(vars, response);
		if (bias)
			reg = LeastSquares.calcCoefficients0(varsReduced, response);
		else
			reg = LeastSquares.calcCoefficients(varsReduced, response);
//		 System.out.println("TEST: " + reg[0] + ", " + reg[reg.length-1]);
//		sortList.add(new PKASort(reg,reducedCols,bias));
//		sortList.add(new EnzymeKineticsSort(reg,reducedCols,bias));
//		sortList.add(new SchloeglSystemProjectedSort(reg,reducedCols,bias));
//		sortList.add(new MAPKSort(reg,reducedCols,bias));

//		sortCoordPtsList.add(1);

		/* BATCH SORT */
//		double[] batchExp = {0.166667,  0.166667,0.333333, 0.333333};
		double[] batchExp = { 0.25, 0.25, 0.25, 0.25 };

//		double[] batchExp = {0.090909, 0.0909091, 0.0909091, 0.0909091, 0.0909091, 0.0909091, 
//				0.0909091, 0.0909091, 0.0909091, 0.0909091, 0.0909091};
//		for (int s = 0; s <  model.numSteps; ++s)
//			sortList.add(new BatchSort<MarkovChainComparable>(batchExp));
//		sortCoordPts = model.dimension();
//		modelDescription = "EnzymeP-batch-sort";

		/* SPLIT SORT */
		for (int s = 0; s < model.K; ++s)
			sortList.add(new SplitSort<MarkovChainComparable>(model.dimension()));
		sortCoordPts = model.dimension();
		modelDescription = "substep-LinearBirthDeath-split-sort";

		/* HILBERT BATCH SORT */
//		sortList.add(new HilbertCurveBatchSort<MarkovChainComparable>(batchExp, 20));
//		sortCoordPtsList.add(1);
//		modelDescription = "EnzymeP-hilberts";

//		sortList.add(new HilbertCurveSort(model.dimension(), 20));
//		sortCoordPtsList.add(1);
//		modelDescription = "cAMP-hilbert-curve";

		ArrayOfComparableChainsSubsteps chain = new ArrayOfComparableChainsSubsteps(model, sortList);

		// MultiDimSort sortPointSet = new SchloeglSystemSort(); //set here if
		// sortCoordPts>1

		StringBuffer sb = new StringBuffer("");
		String str;
		String outFile = modelDescription + ".txt";

		RandomStream stream = new MRG32k3a();
		RQMCPointSet[] rqmcPts;
		PointSet[] pointSets = new PointSet[numSets];
		PointSetRandomization rand;
		RQMCPointSet prqmc;
		int s;

		int nMC = (int) 1E6; // n to estimate MC variance.
		Tally statMC = new Tally();
		statMC.init();
		// model.simulRunsWithSubstreams(nMC, model.numSteps, stream, statMC);
		model.simulRuns(nMC, model.numSteps, stream, statMC);
		model.numSteps /= model.K;
		double varMC = statMC.variance();
		str = "\n\n --------------------------\n";
		str += "MC average  = " + statMC.average() + "\n";
		str += "MC variance = " + varMC + "\n\n";
		sb.append(str);
		System.out.println(str);

		ArrayList<RQMCPointSet[]> listP = new ArrayList<RQMCPointSet[]>();

		// Independent points (Monte Carlo)
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			pointSets[s] = new IndependentPointsCached(N[s], model.K + model.N);
//			rand = new RandomShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("Independent points");
//		listP.add(rqmcPts);
//
//		// Stratification
//		rqmcPts = new RQMCPointSet[numSets];
//		int k;
//		for (s = 0; s < numSets; ++s) {
//			k = (int) Math.round(Math.pow(Num.TWOEXP[s + mink], 1.0 / (double) (sortCoordPts + model.K)));
//			pointSets[s] = new StratifiedUnitCube(k, sortCoordPts + model.K);
//			// Here the points must be sorted at each step, always.
//			// In the case of Hilbert map, the points should be 2d and sorted
//			// based on one coordinate,
//			// whereas the states are 2d and sorted by the Hilbert sort.
//			rand = new RandomShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("Stratification");
//		listP.add(rqmcPts);
//
//		// Lattice + Shift
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			if (sortCoordPts == 1)
//				pointSets[s] = new Rank1Lattice(N[s], a[s], 1 + model.K);
////				 pointSets[s] = new Rank1Lattice(N[s],a,1+model.K);
//			else
//				pointSets[s] = new SortedAndCutPointSet(new Rank1Lattice(N[s], a[s], sortCoordPts + model.K),
//						sortList.get(s));
////					pointSets[s] = new SortedAndCutPointSet (new Rank1Lattice(N[s],a,sortCoordPtsList+model.K),sortList.get(s));
//
//			rand = new RandomShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("lattice+shift");
//		listP.add(rqmcPts);
//
//		// Rank1Lattice +baker
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			if (sortCoordPts == 1)
//				pointSets[s] = new BakerTransformedPointSet(new Rank1Lattice(N[s], a[s], 1 + model.K));
////					pointSets[s] = new BakerTransformedPointSet (new Rank1Lattice(N[s],a,1+model.K));
//
//			else
//				// The points are sorted here, but only once.
//				pointSets[s] = new SortedAndCutPointSet(
//						new BakerTransformedPointSet(new Rank1Lattice(N[s], a[s], sortCoordPts + model.K)),
//						sortList.get(s));
////						 pointSets[s] = new SortedAndCutPointSet (new BakerTransformedPointSet 
////						    		(new Rank1Lattice(N[s],a,sortCoordPtsList+model.K)), sortList.get(s));
//			rand = new RandomShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("lattice+ baker ");
//		listP.add(rqmcPts);

		// Sobol + LMS
		rqmcPts = new RQMCPointSet[numSets];
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new SobolSequence(s + mink, 31, 1 + 1);
			else {

				pointSets[s] = new SortedAndCutPointSet(new SobolSequence(s + mink, 31, sortCoordPts + 1),
						sortList.get(s));
			}
			rand = new LMScrambleShift(stream);
			prqmc = new RQMCPointSet(pointSets[s], rand);
			rqmcPts[s] = prqmc;
		}
		rqmcPts[0].setLabel("Sobol+LMS");
		listP.add(rqmcPts);

		// Sobol + LMS + Baker
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			if (sortCoordPtsList == 1)
//				pointSets[s] = new BakerTransformedPointSet(new SobolSequence(s + mink, 31, 1 + model.K));
//			else
//				pointSets[s] = new SortedAndCutPointSet(
//						new BakerTransformedPointSet(new SobolSequence(s + mink, 31, sortCoordPtsList + model.K)),
//						sortList.get(s));
//
//			rand = new LMScrambleShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("Sobol+LMS+baker");
//		listP.add(rqmcPts);

		// Sobol+NUS
		rqmcPts = new RQMCPointSet[numSets];
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1) {
				CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31, 1 + 1));
				p.setRandomizeParent(false);
				pointSets[s] = p;
			} else {
				CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31, sortCoordPts + 1));
				p.setRandomizeParent(false);
				// The points are sorted here, but only once.
				pointSets[s] = new SortedAndCutPointSet(p, sortList.get(s));
			}
			rand = new NestedUniformScrambling(stream);
			prqmc = new RQMCPointSet(pointSets[s], rand);
			rqmcPts[s] = prqmc;
		}
		rqmcPts[0].setLabel("Sobol+NUS");
		listP.add(rqmcPts);

		for (RQMCPointSet[] ptSeries : listP) {
			String label = ptSeries[0].getLabel();
			str = label;
			str += "\n-----------------------------\n";
			sb.append(str + "\n");
			System.out.println(str);
			// If Stratification, then we need to sort point set in every step
			int sortedCoords = 0;
			str = (chain.testVarianceRateFormat(ptSeries, sortedCoords,  model.numSteps, m, varMC,
					modelDescription + "-" + label, label));
			System.out.println(str);
			sb.append(str + "\n");

		}

		FileWriter file = new FileWriter(outFile);
		file.write(sb.toString());
		file.close();

	}

}
