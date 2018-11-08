package flo.biologyArrayRQMC.examples;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import umontreal.ssj.hups.BakerTransformedPointSet;
import umontreal.ssj.hups.CachedPointSet;
import umontreal.ssj.hups.IndependentPointsCached;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.LatinHypercube;
import umontreal.ssj.hups.NestedUniformScrambling;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.RandomShift;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.hups.SortedAndCutPointSet;
import umontreal.ssj.hups.StratifiedUnitCube;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChainsNN;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.Num;
import umontreal.ssj.util.sort.MultiDimSort;

public class testChemicalReactionNetworkNN {

	public static void main(String[] args) throws IOException {

		ChemicalReactionNetwork model;
		
//		double epsInv = 1E2;
//		double alpha = 1E-4;
//		double[] c = { 1.0, alpha };
//		double[] x0 = { epsInv, epsInv / alpha };
//		double T = 1.6;
//		double tau = 0.2;
//
//		model = new ReversibleIsomerizationComparable(c, x0, tau, T);
//		String modelDescription = "ReversibleIsometrization";
//
//		System.out.println(model.toString());
		
//		double[]c = {3E-7, 1E-4, 1E-3,3.5};
////		double[] x0 = {250.0, 1E5, 2E5};
//		double[] x0 = {250.0,1E5};
//		double N0 = 2E5 + 250.0 + 1E5;
//		double T = 4;
//		double tau = 0.2;
//
//		
//		
//		 model = new SchloeglSystemProjected(c,x0,tau,T,N0);
//		String modelDescription = "SchloeglSystemProj";
		
		double[]c = {8.696E-5, 0.02, 1.154E-4,0.02,0.016,0.0017};//Nano: 1E-9
		double[] x0 = {33000.0,33030.0, 1100.0, 1100.0, 1100.0, 1100.0};
		double T = 0.00005;
		double tau = T/20.0;

		
		
		 model = new PKA(c,x0,tau,T);
		 System.out.println(model.toString());
		 String modelDescription = "PKA";
		
		String dataLabel = "MCData";

		String[] fileNames = new String[model.numSteps];
		for(int s = 0; s < fileNames.length; s++) {

//			fileNames[s] += dataLabel+ "Step" + s + ".zip";

			fileNames[s] = "data/" + modelDescription + "/" + dataLabel+ "Step" + s + ".zip";

			}
		ArrayOfComparableChainsNN chain = new ArrayOfComparableChainsNN(model, fileNames);

		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 }; // n from 8
		// to 20.
//		int[] N = { 512, 1024, 2048, 4096, 8192, 16384,32768, 65536, 131072}; // n from 8

		int[] logN = { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
		int mink = 9;
		int numSets = N.length;
		
		int sortCoordPts = 1;
		MultiDimSort sortPointSet = null; //set here if sortCoordPts>1

		int m = 50;
		
		StringBuffer sb = new StringBuffer("");
		String str;
		String outFile = modelDescription + ".txt";

		RandomStream stream = new MRG32k3a();
		RQMCPointSet[] rqmcPts;
		PointSet[] pointSets = new PointSet[numSets];
		PointSetRandomization rand;
		RQMCPointSet prqmc;
		int s;
		

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

		// Stratification
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

		// Sobol + LMS
		rqmcPts = new RQMCPointSet[numSets];
		for (s = 0; s < numSets; ++s) {
			if (sortCoordPts == 1)
				pointSets[s] = new SobolSequence(s + mink, 31, 1 + model.K);
			else {
				
				pointSets[s] = new SortedAndCutPointSet(new SobolSequence(s + mink, 31, sortCoordPts + model.K),
						sortPointSet);
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
//			if (sortCoordPts == 1)
//				pointSets[s] = new BakerTransformedPointSet(new SobolSequence(s + mink, 31, 1 + model.K));
//			else
//				pointSets[s] = new SortedAndCutPointSet(
//						new BakerTransformedPointSet(new SobolSequence(s + mink, 31, sortCoordPts + model.K)),
//						sortPointSet);
//			// pointSets[s] = new SortedAndCutPointSet(new SobolSequence(s +
//			// mink, 31, sortCoordPts + baseChain.K), sort);
//			rand = new LMScrambleShift(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("Sobol+LMS+baker");
//		listP.add(rqmcPts);

		// Sobol+NUS
//		rqmcPts = new RQMCPointSet[numSets];
//		for (s = 0; s < numSets; ++s) {
//			if (sortCoordPts == 1) {
//				CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31, 1 + model.K));
//				p.setRandomizeParent(false);
//				pointSets[s] = p;
//			} else {
//				CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31, sortCoordPts + model.K));
//				p.setRandomizeParent(false);
//				// The points are sorted here, but only once.
//				pointSets[s] = new SortedAndCutPointSet(p, sortPointSet);
//			}
//			rand = new NestedUniformScrambling(stream);
//			prqmc = new RQMCPointSet(pointSets[s], rand);
//			rqmcPts[s] = prqmc;
//		}
//		rqmcPts[0].setLabel("Sobol+NUS");
//		listP.add(rqmcPts);
		
		

		int nMC = (int) 1E6; // n to estimate MC variance.
		Tally statMC = new Tally();
		statMC.init();
		// model.simulRunsWithSubstreams(nMC, model.numSteps, stream, statMC);
		model.simulRuns(nMC, model.numSteps, stream, statMC);
		double varMC = statMC.variance();
		str = "\n\n --------------------------\n";
		str+="MC average  = " + statMC.average() + "\n";
		str += "MC variance = " + varMC+"\n\n";
		sb.append(str);
		System.out.println(str);
		
		for (RQMCPointSet[] ptSeries : listP) {
			String label = ptSeries[0].getLabel();
			str = label;
			str += "\n-----------------------------\n";
			sb.append(str + "\n");
			System.out.println(str);
			// If Stratification, then we need to sort point set in every step
			int sortedCoords = label.startsWith("St") ? sortCoordPts : 0;
			str = (chain.testVarianceRateFormat(ptSeries, sortedCoords, model.numSteps, m, varMC, modelDescription + label,
					label));
			System.out.println(str);
			sb.append(str + "\n");
		}
		
		FileWriter file = new FileWriter(outFile);
		file.write(sb.toString());
		file.close();
		
		
	}

}
