package flo.biologyArrayRQMC.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

import umontreal.ssj.functionfit.LeastSquares;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.hups.SortedAndCutPointSet;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChains;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChainsMultipleSorts;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChainsNN;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.SplitSort;

public class testChemicalReactionNetworkMultipleSorts {

	public static void main(String[] args) throws IOException {

		ChemicalReactionNetwork model;
		ArrayList<MultiDimSort> sortList = new ArrayList<MultiDimSort>();

		double[]c = {8.696E-5, 0.02, 1.154E-4,0.02,0.016,0.0017};//Nano: 1E-9
		double[] x0 = {33000.0,33030.0, 1100.0, 1100.0, 1100.0, 1100.0};
		double T = 0.00005;
		double tau = T/20.0;
		int numSteps = (int) (T/tau);

		
		
		 model = new PKA(c,x0,tau,T);
		 System.out.println(model.toString());
		 String modelDescription = "PKALessNoise";
		 String dataLabel = "MCDataLessNoise";
		String sortLabel = "linear";
			
		int rows = 32768;
		int cols = 7;

		
		Scanner sc;
		double[][] vars = new double[rows][cols-1];
		double[] response = new double[rows];
		double[] reg;
			

				
		for (int s = 1; s < numSteps; s++) {
			sc = new Scanner( new BufferedReader(new FileReader("data/PKA/MCDataLessNoise_Step_" + s + ".csv")));
			for (int i=0; i<rows; i++) {
	            String[] line = sc.nextLine().trim().split(",");
	            response[i] = Double.parseDouble(line[cols-1]);
	            for (int j=0; j<cols-1; j++) {
	              vars[i][j] =Double.parseDouble(line[j]);
	            }
	         }
			 reg = LeastSquares.calcCoefficients0(vars, response);
//			 System.out.println("TEST: " + reg[0] + ", " + reg[reg.length-1]);
			sortList.add(new PKASort(reg));
			if(s==1) {
				sortList.add(new PKASort(reg));
//				 System.out.println("TEST: JEP!");
			}
		}
		
		
		
/*		double[] c = { 3E-7, 1E-4, 1E-3, 3.5 };
		double[] x0 = { 250.0, 1E5 };
		double N0 = 2E5 + 250.0 + 1E5;
		double T = 4;
		double tau = 0.2;

		model = new SchloeglSystemProjected(c, x0, tau, T,N0);
		String modelDescription = "SchloeglSystemProj";
		String dataLabel = "MCData";
		String sortLabel = "l1";

		ArrayList<MultiDimSort> sortList = new ArrayList<MultiDimSort>();
	double[]	a = {258.08424900431476, 258.08424900431476, 261.48079476115794, 
				   263.4428886265869, 264.8616180800489, 266.305793668144, 
				   267.53304006024337, 268.6651710041849, 269.69495593902553, 
				   270.53027085596557, 271.3720774552504, 272.102275628243, 
				   272.59645204814956, 273.06747203948294, 273.5221092642125, 
				   273.9659256193167, 274.40425544095166, 274.9585940584114, 
				   275.57259995678515, 276.6022004917788};
	double[]		b = {-183.80616601113414, -183.80616601113414, -182.607100941182, 
				-183.0113463601882, -182.60033595073241, -182.74224034886402, 
				-183.03298944270117, -183.3026018888793, -183.7340529921913, 
				-184.47510841363902, -185.3685660345213, -186.5548323910866, 
				-188.20757076613, -190.6684160043247, -194.10142830340257, 
				-199.6931961856299, -208.5797950130828, -223.3778391917219, 
				-251.23969976999376, -320.57268446165193 };
	double[]		cc = {-0.020718048328757437, -0.020718048328757437, 
				-0.02085665892524057, -0.02050430972354262, -0.020321468483031433, 
				-0.01994455249078135, -0.0193665061750056, -0.01889693750412154, 
				-0.018155375927028216, -0.01719047333196788, -0.016191738209059798, 
				-0.015095729923674129, -0.01383336327857643, -0.012552046103108346, 
				-0.011278578982519468, -0.009914829637951084, -0.00850287287400984, 
				-0.007041211544763277, -0.0054769292061915996, -0.0037027507157621027};
	double[]	d = {-0.0004121763550424434, -0.0004121763550424434, 
				-0.0003172872596619393, -0.0002984301587111952, 
				-0.00030416451356721127, -0.0002894964829811772, 
				-0.00026188343798948103, -0.00028320414914643414, 
				-0.0002562027307717224, -0.00022520036419196947, 
				-0.00020498750823635462, -0.00018941651036391923, 
				-0.00016077589696169295, -0.0001475501369799153, 
				-0.0001330120153858751, -0.00011591745302139875, 
				-0.00009408579516639558, -0.00006673785744387753, 
				-0.00003785054779129977, -0.000013611715811696467};
	double[]	e = {46.44815806307885, 46.44815806307885, 37.00570410155215, 
				   35.02706326387889, 35.537353612864536, 33.969420886161004, 
				   31.066626075015538, 33.04259381385883, 30.161600693699768, 
				   26.828897216859286, 24.556321027528586, 22.717335558885758, 
				   19.547647654439107, 17.890555286673614, 16.108003871244392, 
				   14.051956607085222, 11.522227231467363, 8.444783296776354, 
				   5.195040493861866, 2.3474355837984646};
		for(int i = 0; i < a.length; i++) {
//			sortList.add(new SchloeglSystemProjectedSort(a[i],b[i],cc[i],d[i],e[i]));
			sortList.add(new ProjectedL1Sort(model.X0,c,(double)i,N0));
		}
*/
		ArrayOfComparableChainsMultipleSorts chain = new ArrayOfComparableChainsMultipleSorts(model,sortList);

//		 int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
//		 524288, 1048576 }; // n from 8
		// to 20.
		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}; // n from 8
		

		int[] logN = { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
		int mink = 18;
		int numSets = N.length;
		numSets = 3;

		int sortCoordPtsList = 1;
	
		
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
		int s;

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
				if (sortCoordPtsList == 1)
					pointSets[s] = new SobolSequence(s + mink, 31, 1 + model.K);
				else {

					pointSets[s] = new SortedAndCutPointSet(
							new SobolSequence(s + mink, 31, 1 + model.K), sortList.get(19));
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
				int sortedCoords =  0;
				str = (chain.testVarianceRateFormat(ptSeries, sortedCoords, model.numSteps, m, varMC,
						modelDescription+ "-" +sortLabel+ "-" + label, label));
				System.out.println(str);
				sb.append(str + "\n");

			}
			
		FileWriter file = new FileWriter(outFile);
		file.write(sb.toString());
		file.close();

	}

}
