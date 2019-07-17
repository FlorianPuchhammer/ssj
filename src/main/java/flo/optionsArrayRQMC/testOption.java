package flo.optionsArrayRQMC;

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
import umontreal.ssj.hups.KorobovLattice;
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
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.Num;
import umontreal.ssj.util.sort.BatchSort;
import umontreal.ssj.util.sort.HilbertCurveBatchSort;
import umontreal.ssj.util.sort.HilbertCurveSort;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.MultiDimSortComparable;
import umontreal.ssj.util.sort.SplitSort;

public class testOption {

	public static void main(String[] args) throws IOException {

//		MarkovChainComparable model;
		ArrayList<Integer> sortCoordPtsList = new ArrayList<Integer>();
		ArrayList<MultiDimSort> sortList = new ArrayList<MultiDimSort>();
		ArrayList<MultiDimSort> sortPts = new ArrayList<MultiDimSort>();

		String modelDescription;
		/*
		 * ******************* VGA
		 **********************/
		
		int d = 10;
		double t1 = (240.0 - d + 1) / 365.0;

		double T = 240.0 / 365.0;
		double K = 100.0;
		double s0 = 100.0;
		double r = 0.1;
		double nu = 0.3;
		double teta = -0.1436;
		double sigma = 0.12136;
		double muu = 1.0;

		 VGAsianOptionComparable model = new VGAsianOptionComparable (r, d, t1, T, K,nu,muu,teta,s0, sigma, new VGAsianOptionComparable.NormalCDFMap());
		int dim = 2; // #pts needed to advance chain
		
		modelDescription = "VarianceGamma";
		
		/*
		 * ******************* HESTON
		 **********************/
		
//		double r = 0.05;		
//		double T = 1;
//		int d = 16;
//		double K = 100.0;
//		double S0 = 100.0;
//		double V0 = 0.04;
//		double sigma = 0.2;
//		double lambda=5;
//		double xi=0.25;
//		double correlation= -0.5;
//		modelDescription = "HestonModel";
//		 model = new HestonModelComparablef(r,sigma,T,d,lambda,xi,S0,V0,K,correlation,new HestonModelComparablef.NormalCDFMap());
//		int dim =2;
		
		System.out.println(model.toString());
		
		ArrayOfComparableChains chain = new ArrayOfComparableChains(model);
		
		
		int numSteps = d;


//		String modelDescription = "VG" + "-split-" + splitter + "-"+(numSteps-splitter);


		

		int[] N = { 65536, 131072, 262144, 524288, 1048576 }; // n from 8
		// to 20.
		// int[] N = { 262144,
		// 524288, 1048576}; // n from 8

		int[] logN = {  16, 17, 18, 19, 20 };

		int[][] a = {

				{1, 19463, 8279, 14631, 12629, 3257},

				{1, 50687, 44805, 12937, 21433, 42925},

				{1, 100135, 28235, 39865, 43103, 121135},

				{1, 154805, 242105, 171449, 27859, 174391},

				{1, 387275, 314993, 50301, 174023, 354905}

		};
		
		int[] korA = { 38295, 114789, 177473, 286857, 271251};

		int mink = 16;
		int numSets = N.length;



		int m = 100;  //m=20; //goto

		String sortLabel;
		
		/* BATCH SORT */
//		double[] batchExp = {0.166667,  0.166667,0.333333, 0.333333};
		double[] batchExp = { 0.5, 0.5 };

		sortList.add(new BatchSort<MarkovChainComparable>(batchExp));
		sortPts.add(new BatchSort<MarkovChainComparable>(batchExp));
		sortCoordPtsList.add(model.dimension());
		sortLabel = "batch-sort";

		/* SPLIT SORT */
//		sortList.add(new SplitSort<MarkovChainComparable>(model.dimension()));
//		sortCoordPtsList.add(model.dimension());
//		modelDescription = "LinBirthDeath-split-sort";

		/* HILBERT BATCH SORT */
//		sortList.add(new HilbertCurveBatchSort<MarkovChainComparable>(batchExp, 20));
//		sortPts.add(new SplitSort<MarkovChainComparable>(1));
//		sortCoordPtsList.add(1);
//		sortLabel = "HilbertBatch";

//		sortList.add(new HilbertCurveSort(model.getStateDimension(), 20));
//		sortCoordPtsList.add(1);
//		sortPts.add(new SplitSort<MarkovChainComparable>(1));
//		sortLabel = "hilbert-curve";
		
	
		
		StringBuffer sb = new StringBuffer("");
		String str;
		String outFile = modelDescription + "-" + sortLabel + ".txt";

		RandomStream stream = new MRG32k3a();
		RQMCPointSet[] rqmcPts;
		PointSet[] pointSets = new PointSet[numSets];
		PointSetRandomization rand;
		RQMCPointSet prqmc;
		int s;

//		System.out.println(model.toString());
		
		int nMC = (int) 1E6; // n to estimate MC variance.
//		nMC = (int)1E2;
//		m=10;
//		N=new int[] {128,256,512};
//		mink=7;
		Tally statMC = new Tally();
		statMC.init();
		// model.simulRunsWithSubstreams(nMC, model.numSteps, stream, statMC);

		model.simulRuns(nMC, d, stream, statMC);
		double varMC = statMC.variance();
		str = "\n\n --------------------------\n";
		str += "MC average  = " + statMC.average() + "\n";
		str += "MC variance = " + varMC + "\n\n";
		sb.append(str);
		System.out.println(str);

		int i = 0; // Sorts indexed by i
		for (MultiDimSort sort : sortList) {
			str = "****************************************************\n";
			str += "*\t" + sort.toString() + "\n";
			str += "****************************************************\n\n";
			sb.append(str);
			System.out.println(str);
			ArrayList<RQMCPointSet[]> listP = new ArrayList<RQMCPointSet[]>();
		
		System.out.println("Sort: " + sortLabel);

		
		// Independent points (Monte Carlo)
//		 rqmcPts = new RQMCPointSet[numSets];
//		 for (s = 0; s < numSets; ++s) {
//		 pointSets[s] = new IndependentPointsCached(N[s], sortCoordPtsList.get(i)+ dim);
//		 rand = new RandomShift(stream);
//		 prqmc = new RQMCPointSet(pointSets[s], rand);
//		 rqmcPts[s] = prqmc;
//		 }
//		 rqmcPts[0].setLabel("Independent points");
//		 listP.add(rqmcPts);

		// Stratification
//			rqmcPts = new RQMCPointSet[numSets];
//			int k;
//			for (s = 0; s < numSets; ++s) {
//				k = (int) Math.round(Math.pow(Num.TWOEXP[s + mink], 1.0 / (double) (sortCoordPtsList.get(i) + dim)));
//				pointSets[s] = new StratifiedUnitCube(k, sortCoordPtsList.get(i) + dim);
//				// Here the points must be sorted at each step, always.
//				// In the case of Hilbert map, the points should be 2d and sorted
//				// based on one coordinate,
//				// whereas the states are 2d and sorted by the Hilbert sort.
//				rand = new RandomShift(stream);
//				prqmc = new RQMCPointSet(pointSets[s], rand);
//				rqmcPts[s] = prqmc;
//			}
//			rqmcPts[0].setLabel("Stratification");
//			listP.add(rqmcPts);

		// Lattice + Shift
//		rqmcPts = new RQMCPointSet[numSets];
//		 for (s = 0; s < numSets; ++s){
//			
//				pointSets[s] = new SortedAndCutPointSet (new Rank1Lattice(N[s],a[s],sortCoordPtsList.get(i)+dim),sortPts.get(i));
//
//				 rand = new RandomShift(stream);
//			 prqmc = new RQMCPointSet(pointSets[s], rand);
//			 rqmcPts[s] = prqmc;
//		 }
//		 rqmcPts[0].setLabel("lattice+shift");
//		 listP.add(rqmcPts);

		// Rank1Lattice +baker
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			
//				pointSets[s] = new SortedAndCutPointSet(new BakerTransformedPointSet(
//						new Rank1Lattice(N[s], a[s], sortCoordPtsList.get(i) + dim)), sortPts.get(i));
//			rand = new RandomShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("lattice+ baker ");
//		listP.add(rqmcPts);

		// Sobol + LMS
		rqmcPts = new RQMCPointSet[numSets];
		for (s = 0; s < numSets; ++s) {
			
			

				pointSets[s] = new SortedAndCutPointSet(
						new SobolSequence(s + mink, 31, sortCoordPtsList.get(i) + dim), sortPts.get(i));
			
			rand = new LMScrambleShift(stream);
			prqmc = new RQMCPointSet(pointSets[s], rand);
			rqmcPts[s] = prqmc;
		}
		rqmcPts[0].setLabel("Sobol+LMS");
		listP.add(rqmcPts);

		// Sobol + LMS + Baker
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			if (sortCoordPtsList.get(i) == 1)
//				pointSets[s] = new BakerTransformedPointSet(new SobolSequence(s + mink, 31, 1 + model.K));
//			else
//				pointSets[s] = new SortedAndCutPointSet(
//						new BakerTransformedPointSet(new SobolSequence(s + mink, 31, sortCoordPtsList.get(i) + model.K)),
//						sortList.get(i));
//
//			rand = new LMScrambleShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("Sobol+LMS+baker");
//		listP.add(rqmcPts);

		// Sobol+NUS
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			
//				CachedPointSet p = new CachedPointSet(
//						new SobolSequence(s + mink, 31, sortCoordPtsList.get(i) + dim));
//				p.setRandomizeParent(false);
//				// The points are sorted here, but only once.
//				pointSets[s] = new SortedAndCutPointSet(p, sortPts.get(i));
//			
//			rand = new NestedUniformScrambling(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("Sobol+NUS");
//		listP.add(rqmcPts);
		 
		 for (RQMCPointSet[] ptSeries : listP) {
				String label = ptSeries[0].getLabel();
				str = label;
				str += "\n-----------------------------\n";
				sb.append(str + "\n");
				System.out.println(str);
				// If Stratification, then we need to sort point set in every step
				int sortedCoords = label.startsWith("St") ? sortCoordPtsList.get(i) : 0;
				str = (chain.testVarianceRateFormat(ptSeries, sort, sortedCoords, model.numSteps, m, varMC,
						modelDescription + "-" + sort.toString() + "-" + label, label));
				System.out.println(str);
				sb.append(str + "\n");

//				System.out.println(model.toString());
			}
			i++;

		}

		FileWriter file = new FileWriter(outFile);
		file.write(sb.toString());
		file.close();

	}

}
