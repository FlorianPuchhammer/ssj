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
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.MultiDimSortComparable;
import umontreal.ssj.util.sort.SplitSort;

public class testOptionMultipleSorts {

	public static void main(String[] args) throws IOException {

		MarkovChainComparable model;
		ArrayList<MultiDimSort> sortList = new ArrayList<MultiDimSort>();
		MultiDimSort sortPts ;

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

		 model = new VGAsianOptionComparable (r, d, t1, T, K,nu,muu,teta,s0, sigma, new VGAsianOptionComparable.NormalCDFMap());
		int dim = 2; // #pts needed to advance chain
		
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
//		
//		 model = new HestonModelComparablef(r,sigma,T,d,lambda,xi,S0,V0,K,correlation,new HestonModelComparablef.NormalCDFMap());
//		int dim =2;
		
		
		int numSteps = d;
		int splitter = d;
		int sortCoordPtsList=1;
//		int[] batches =  {1};
//		sortPts = new BatchSort(batches);
		sortPts = new SplitSort(sortCoordPtsList);
		double aa = 0.5;
		double bb = 1.0 - aa;
		System.out.println(model.toString());
//		String modelDescription = "VG" + "-split-" + splitter + "-"+(numSteps-splitter);
		String modelDescription = "VG" + "-linear";
		String sortLabel = "a" + aa + "-b" + bb;
		
//		double[] ptBatchExp1 = {0.33,0.33,0.34};
//		double[] ptBatchExp2 = {0.5,0.5};
//		MultiDimSortComparable<MarkovChainComparable> ptSort = new BatchSort<MarkovChainComparable>(ptBatchExp2);
		MultiDimSortComparable<MarkovChainComparable> ptSort = new SplitSort<MarkovChainComparable>(2);

		for(int s = 0; s < splitter; s++) {
			double fac = (double)s/(double)(numSteps-1);
//			aa = 1.0 - fac ;
			aa=0.5;
//			aa = fac;
//			aa = 1.0 - Math.sqrt(fac) ; 
//			aa = (1+r/(double)(numSteps));
			bb = 1.0 -aa;
//			bb = s;
			
			sortList.add(new VGLinearSort(aa,bb));
		
//			sortList.add(new HestonModelMixedSort(aa,bb,ptSort));
//			sortList.add(new HestonModelOneDimSort(aa,bb,0.5,numSteps));
			
//				sortList.add(new BatchSort<MarkovChainComparable>(ptBatchExp1));
//				sortList.add(new SplitSort<MarkovChainComparable>(3) );
//				sortList.add(new VGLinearSort(aa,bb));
				
		
		}

		for (int s = splitter; s < numSteps; s++) {
			sortList.add(new VGLinearSort(aa,bb));
//			sortList.add(new HestonModelMixedSort(bb,aa));
		}

		ArrayOfComparableChainsMultipleSorts chain = new ArrayOfComparableChainsMultipleSorts(model, sortList);

		int[] N = { 65536, 131072, 262144, 524288, 1048576 }; // n from 8
//		int[] N = {512, 1024, 2048, 4096, 8192};
		int mink = 16; //mink = 9;
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

		
		int numSets = N.length;
//		numSets = 3;

	

		// MultiDimSort sortPointSet = new SchloeglSystemSort(); //set here if
		// sortCoordPts>1

		int m = 100;  //m=20; //goto


		StringBuffer sb = new StringBuffer("");
		String str;
		String outFile = modelDescription + ".txt";

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
		
		System.out.println("Sort: " + sortLabel);

		ArrayList<RQMCPointSet[]> listP = new ArrayList<RQMCPointSet[]>();

		
		
		// Independent points (Monte Carlo)
		 rqmcPts = new RQMCPointSet[numSets];
		 for (s = 0; s < numSets; ++s) {
		 pointSets[s] = new IndependentPointsCached(N[s], sortCoordPtsList+ dim);
		 rand = new RandomShift(stream);
		 prqmc = new RQMCPointSet(pointSets[s], rand);
		 rqmcPts[s] = prqmc;
		 }
		 rqmcPts[0].setLabel("Independent points");
		 listP.add(rqmcPts);
		 
//		// Stratification
//			rqmcPts = new RQMCPointSet[numSets];
//			int k;
//			for (s = 0; s < numSets; ++s) {
//				k = (int) Math.round(Math.pow(Num.TWOEXP[s + mink], 1.0 / (double) (sortCoordPtsList + dim)));
//				pointSets[s] = new StratifiedUnitCube(k, sortCoordPtsList + dim);
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

		//Lattice + Shift
//		rqmcPts = new RQMCPointSet[numSets];
//		 for (s = 0; s < numSets; ++s){
//			 
//				pointSets[s] = new SortedAndCutPointSet (new Rank1Lattice(N[s],a[s],sortCoordPtsList+dim),sortPts);
////					pointSets[s] = new SortedAndCutPointSet (new KorobovLattice(N[s],korA[s],sortCoordPtsList+dim),sortList.get(s));
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
//			// The points are sorted here, but only once.
//			pointSets[s] = new SortedAndCutPointSet(
//					new BakerTransformedPointSet(new Rank1Lattice(N[s], a[s], sortCoordPtsList + dim)), sortPts);
//
////						 pointSets[s] = new SortedAndCutPointSet (new BakerTransformedPointSet 
////						    		(new KorobovLattice(N[s],korA[s],sortCoordPtsList+dim)), sortList.get(s));
//			rand = new RandomShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("lattice+ baker ");
//		listP.add(rqmcPts);

		 		
		// Sobol + LMS
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			
//				pointSets[s] = new SortedAndCutPointSet(new SobolSequence(s + mink, 31, sortCoordPtsList+ dim), sortPts);
////				((SortedAndCutPointSet)pointSets[s]).sortByCoordinate(0);
////				pointSets[s] = new SortedAndCutPointSet(new SobolSequence(s + mink, 31, sortCoordPtsList+ dim),ptSort);
//
//			
//			rand = new LMScrambleShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("Sobol+LMS");
//		listP.add(rqmcPts);

		// Sobol + LMS + Baker
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			if (sortCoordPtsList== 1)
//				pointSets[s] = new BakerTransformedPointSet(new SobolSequence(s + mink, 31, 1 + dim));
//			else
//				pointSets[s] = new SortedAndCutPointSet(
//						new BakerTransformedPointSet(new SobolSequence(s + mink, 31, sortCoordPtsList +dim)),
//						sortList.get(s));
//
//			rand = new LMScrambleShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("Sobol+LMS+baker");
//		listP.add(rqmcPts);

		// Sobol+NUS
//		 rqmcPts = new RQMCPointSet[numSets];
//		 for (s = 0; s < numSets; ++s) {
//		
//		 CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31,
//				 sortCoordPtsList + dim));
//		 p.setRandomizeParent(false);
//		 // The points are sorted here, but only once.
//		 pointSets[s] = new SortedAndCutPointSet(p, sortPts);
////		 pointSets[s] = new SortedAndCutPointSet(p,  ptSort);
//		 
//		 rand = new NestedUniformScrambling(stream);
//		 prqmc = new RQMCPointSet(pointSets[s], rand);
//		 rqmcPts[s] = prqmc;
//		 }
//		 rqmcPts[0].setLabel("Sobol+NUS");
//		 listP.add(rqmcPts);
		 
		for (RQMCPointSet[] ptSeries : listP) {
			String label = ptSeries[0].getLabel();
			str = label;
			str += "\n-----------------------------\n";
			sb.append(str + "\n");
			System.out.println(str);
			// If Stratification, then we need to sort point set in every step
			int sortedCoords = label.startsWith("St") ? sortCoordPtsList : 0;
			str = (chain.testVarianceRateFormat(ptSeries, sortedCoords, d, m, varMC,
					modelDescription + "-" +sortLabel + "-" + label, label));
			System.out.println(str);
			sb.append(str + "\n");

		}

		FileWriter file = new FileWriter(outFile);
		file.write(sb.toString());
		file.close();

	}

}
