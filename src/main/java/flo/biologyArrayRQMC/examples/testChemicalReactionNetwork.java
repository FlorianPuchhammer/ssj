package flo.biologyArrayRQMC.examples;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.hups.SortedAndCutPointSet;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChains;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChainsNN;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.sort.HilbertCurveSort;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.SplitSort;

public class testChemicalReactionNetwork {

	public static void main(String[] args) throws IOException {

		ChemicalReactionNetwork model;

		// double epsInv = 1E2;
		// double alpha = 1E-4;
		// double[] c = { 1.0, alpha };
		// double[] x0 = { epsInv, epsInv / alpha };
		// double T = 1.6;
		// double tau = 0.2;
		//
		// model = new ReversibleIsomerizationComparable(c, x0, tau, T);
		// String modelDescription = "ReversibleIsometrization";
		//
		// System.out.println(model.toString());

//		double[] c = { 3E-7, 1E-4, 1E-3, 3.5 };
//		double[] x0 = { 250.0, 1E5, 2E5 };
//		double T = 4;
//		double tau = 0.2;
//
//		model = new SchloeglSystem(c, x0, tau, T);
//		String modelDescription = "SchloeglSystem";
		
//		double[] c = { 3E-7, 1E-4, 1E-3, 3.5 };
//		double[] x0 = { 250.0, 1E5 };
//
//		double N0 = 2E5 + 250.0 + 1E5;
//		double T = 4;
//
//		double tau = 0.2;
		
//		model = new SchloeglSystemProjected(c, x0, tau, T,N0);
//		String modelDescription = "SchloeglSystemProj";
//		String dataLabel = "MCData";
		
		double[]c = {8.696E-5, 0.02, 1.154E-4,0.02,0.016,0.0017};//Nano: 1E-9
		double[] x0 = {33000.0,33030.0, 1100.0, 1100.0, 1100.0, 1100.0};
		double T = 0.00005;
		double tau = T/20.0;

		
		
		 model = new PKA(c,x0,tau,T);
		 System.out.println(model.toString());
		 String modelDescription = "PKALessNoise";
//		String dataFolder = "data/PKA/";
		model.init();

		ArrayOfComparableChains chain = new ArrayOfComparableChains(model);

//		 int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
//		 524288, 1048576 }; // n from 8
		// to 20.
		int[] N = {262144,
				 524288, 1048576}; // n from 8
		

		int[] logN = { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
		int mink = 18;
		int numSets = N.length;

		ArrayList<Integer> sortCoordPtsList = new ArrayList<Integer>();
		ArrayList<MultiDimSort> sortList = new ArrayList<MultiDimSort>();
		
//		sortList.add(new SchloeglSystemProjectedSort(276.6022004917788,-320.57268446165193, -0.0037027507157621027,-0.000013611715811696467,2.3474355837984646));
//		sortCoordPtsList.add(1);
		//normal PKASort
//		sortList.add(new PKASort(4001.843189480257,0.187482040311803,0.5491689981787978,0.33930041742527806,1.4881627893660272,1.8965252334871867,-0.20397563983883307));
//		sortCoordPtsList.add(1);
		//less-noise PKASort
		sortList.add(new PKASort(206.19660957631277,0.996641406322496,-0.025224064624056866,0.0004189966548408367,0.0013143537810302761,0.15111412538331995,0.0837105163740771));
		sortCoordPtsList.add(1);
//		sortList.add(new SplitSort<MarkovChainComparable>(6));
//		sortCoordPtsList.add(6);
//		sortList.add(new HilbertCurveSort(6, 12));
//		sortCoordPtsList.add(6);
		
		
		// MultiDimSort sortPointSet = new SchloeglSystemSort(); //set here if
		// sortCoordPts>1

		int m = 50;


		StringBuffer sb = new StringBuffer("");
		String str;
		String outFile = modelDescription + ".txt";

		RandomStream stream = new MRG32k3a();
		RQMCPointSet[] rqmcPts;
		PointSet[] pointSets = new PointSet[numSets];
		PointSetRandomization rand;
		RQMCPointSet prqmc;
		int i, s;

		int nMC = (int) 1E6; // n to estimate MC variance.
		Tally statMC = new Tally();
		statMC.init();
		// model.simulRunsWithSubstreams(nMC, model.numSteps, stream, statMC);
		model.simulRuns(nMC, model.numSteps, stream, statMC);
		double varMC = statMC.variance();
		str = "\n\n --------------------------\n";
		str += "MC average  = " + statMC.average() + "\n";
		str += "MC variance = " + varMC + "\n\n";
		sb.append(str);
		System.out.println(str);

		i = 0; // Sorts indexed by i
		for (MultiDimSort sort : sortList) {
			str = "****************************************************\n";
			str += "*\t" + sort.toString() + "\n";
			str += "****************************************************\n\n";
			sb.append(str);
			System.out.println(str);
			ArrayList<RQMCPointSet[]> listP = new ArrayList<RQMCPointSet[]>();

			// Independent points (Monte Carlo)
			// rqmcPts = new RQMCPointSet[numSets];
			// for (s = 0; s < numSets; ++s) {
			// pointSets[s] = new IndependentPointsCached(N[s], model.K + model.N);
			// rand = new RandomShift(stream);
			// prqmc = new RQMCPointSet(pointSets[s], rand);
			// rqmcPts[s] = prqmc;
			// }
			// rqmcPts[0].setLabel("Independent points");
			// listP.add(rqmcPts);

			// Stratification
			// rqmcPts = new RQMCPointSet[numSets];
			// int k;
			// for (s = 0; s < numSets; ++s) {
			// k = (int) Math.round(Math.pow(Num.TWOEXP[s + mink], 1.0 / (double)
			// (sortCoordPts + model.K)));
			// pointSets[s] = new StratifiedUnitCube(k, sortCoordPts + model.K);
			// // Here the points must be sorted at each step, always.
			// // In the case of Hilbert map, the points should be 2d and sorted
			// // based on one coordinate,
			// // whereas the states are 2d and sorted by the Hilbert sort.
			// rand = new RandomShift(stream);
			// prqmc = new RQMCPointSet(pointSets[s], rand);
			// rqmcPts[s] = prqmc;
			// }
			// rqmcPts[0].setLabel("Stratification");
			// listP.add(rqmcPts);

			// Sobol + LMS
			rqmcPts = new RQMCPointSet[numSets];
			for (s = 0; s < numSets; ++s) {
				if (sortCoordPtsList.get(i) == 1)
					pointSets[s] = new SobolSequence(s + mink, 31, 1 + model.K);
				else {

					pointSets[s] = new SortedAndCutPointSet(
							new SobolSequence(s + mink, 31, sortCoordPtsList.get(i) + model.K), sort);
				}
				rand = new LMScrambleShift(stream);
				prqmc = new RQMCPointSet(pointSets[s], rand);
				rqmcPts[s] = prqmc;
			}
			rqmcPts[0].setLabel("Sobol+LMS");
			listP.add(rqmcPts);

			// Sobol + LMS + Baker
			// rqmcPts = new RQMCPointSet[numSets];
			// for (s = 0; s < numSets; ++s) {
			// if (sortCoordPts == 1)
			// pointSets[s] = new BakerTransformedPointSet(new SobolSequence(s + mink, 31, 1
			// + model.K));
			// else
			// pointSets[s] = new SortedAndCutPointSet(
			// new BakerTransformedPointSet(new SobolSequence(s + mink, 31, sortCoordPts +
			// model.K)),
			// sortPointSet);
			// // pointSets[s] = new SortedAndCutPointSet(new SobolSequence(s +
			// // mink, 31, sortCoordPts + baseChain.K), sort);
			// rand = new LMScrambleShift(stream);
			// prqmc = new RQMCPointSet(pointSets[s], rand);
			// rqmcPts[s] = prqmc;
			// }
			// rqmcPts[0].setLabel("Sobol+LMS+baker");
			// listP.add(rqmcPts);

			// Sobol+NUS
			// rqmcPts = new RQMCPointSet[numSets];
			// for (s = 0; s < numSets; ++s) {
			// if (sortCoordPts == 1) {
			// CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31, 1 +
			// model.K));
			// p.setRandomizeParent(false);
			// pointSets[s] = p;
			// } else {
			// CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31,
			// sortCoordPts + model.K));
			// p.setRandomizeParent(false);
			// // The points are sorted here, but only once.
			// pointSets[s] = new SortedAndCutPointSet(p, sortPointSet);
			// }
			// rand = new NestedUniformScrambling(stream);
			// prqmc = new RQMCPointSet(pointSets[s], rand);
			// rqmcPts[s] = prqmc;
			// }
			// rqmcPts[0].setLabel("Sobol+NUS");
			// listP.add(rqmcPts);

			for (RQMCPointSet[] ptSeries : listP) {
				String label = ptSeries[0].getLabel();
				str = label;
				str += "\n-----------------------------\n";
				sb.append(str + "\n");
				System.out.println(str);
				// If Stratification, then we need to sort point set in every step
				int sortedCoords = label.startsWith("St") ? sortCoordPtsList.get(i) : 0;
				str = (chain.testVarianceRateFormat(ptSeries, sort, sortedCoords, model.numSteps, m, varMC,
						modelDescription+ "-" +sort.toString()+ "-" + label, label));
				System.out.println(str);
				sb.append(str + "\n");

			}
			i++;
		}
		FileWriter file = new FileWriter(outFile);
		file.write(sb.toString());
		file.close();

	}

}
