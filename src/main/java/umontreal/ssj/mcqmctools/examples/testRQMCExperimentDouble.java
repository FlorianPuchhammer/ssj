package umontreal.ssj.mcqmctools.examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import umontreal.ssj.hups.BakerTransformedPointSet;
import umontreal.ssj.hups.CachedPointSet;
import umontreal.ssj.hups.FaureSequence;
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
import umontreal.ssj.latnetbuilder.DigitalNetBase2FromLatNetBuilder;
import umontreal.ssj.latnetbuilder.Rank1LatticeFromLatNetBuilder;
import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.mcqmctools.RQMCExperimentSeries;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.PgfDataTable;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.Num;

public class testRQMCExperimentDouble {

	public static void main(String[] args) throws IOException {
		MonteCarloModelDouble model;
		String modelDescr;
		RandomStream noise = new MRG32k3a();
		int dim;
//		String outdir = "/u/puchhamf/misc/jars/creditMetrics/data/creditMetrics/KP-trial/";
		String outdir = "";
		FileWriter fw;
		File file;
		StringBuffer sb = new StringBuffer("");

//		String filename = "KP-trial.dat";
//		model = new CreditMetrics(outdir + filename, noise);
//		double norma = ((CreditMetrics) model).nom();
//		((CreditMetrics) model).normalize(norma);
//		modelDescr = model.toString();
//		dim = ((CreditMetrics) model).getDimension();
//		
//		sb.append( ((CreditMetrics) model).toStringHeader());
//		String str = sb.toString();
//		System.out.println(str);

//		double a = 0.5;
//		dim = 5;
//		model = new GFunction(a, dim);
//		sb.append(model.toString());
//		modelDescr = "GFunction";
//		System.out.println(sb.toString());
		
		double a = 2;
		dim = 5;
		double u = 0.5;
		model = new GenzGaussianPeak(a, u,dim);
		sb.append(model.toString());
		modelDescr = "GenzGaussianPeak";
		System.out.println(sb.toString());


		// Define the RQMC point sets to be used in experiments.
		int basis = 2; // Basis for the loglog plots.
		int numSkipReg = 0; // Number of sets skipped for the regression.
		int mink = 13; // first log(N) considered
		int i;
		int m = 500; // Number of RQMC randomizations.
//		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152 }; // 13
		int[] N = { 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 };
		int numSets = N.length; // Number of sets in the series.

		// Create a list of series of RQMC point sets.
		ArrayList<RQMCPointSet[]> listRQMC = new ArrayList<RQMCPointSet[]>();
		PointSet p;
		PointSetRandomization rand;
		RQMCPointSet[] rqmcPts;
		String[] merits;
		String path;

//		// Independent points (Monte Carlo)
//		rqmcPts = new RQMCPointSet[numSets];
//		for (i = 0; i < numSets; ++i) {
//			p = new IndependentPointsCached(N[i], dim);
//			rand = new RandomShift(noise);
//			rqmcPts[i] = new RQMCPointSet(p, rand);
//		}
//		rqmcPts[0].setLabel("Independent points");
//		listRQMC.add(rqmcPts);
//
////		ORDINARY LATTICE
////		 merits = new String[] {"P2","R0p2","R1","R2","spectral"};
//		 merits = new String[] {"P2","P4","P6","R1","R2","R4","R6","spectral"};
//
//		 path = "/u/puchhamf/misc/latnetbuilder/output/GenzGaussianPeak/d5a2u0p5/ord/0p05k/rank1/";
//		
//		//Lat+Shift
//		for(String merit : merits) {
//			 rqmcPts = new RQMCPointSet[numSets];
//			 for (i = 0; i < numSets; ++i) {
//			
//			 p = new Rank1LatticeFromLatNetBuilder(path + merit + "/2^" +(mink+i) + "/outputMachine.txt" ,dim);
//			
//			 rand = new RandomShift(noise);
//			 rqmcPts[i] = new RQMCPointSet(p, rand);
//			 }
//			 rqmcPts[0].setLabel("Lattice+Shift+"+merit);
//			 listRQMC.add(rqmcPts);
//		}
//		
//		//Lat+Baker
//		for(String merit : merits) {
//			 rqmcPts = new RQMCPointSet[numSets];
//			 for (i = 0; i < numSets; ++i) {
//			
//			 p =  new BakerTransformedPointSet(new Rank1LatticeFromLatNetBuilder(path + merit + "/2^" +(mink+i) + "/outputMachine.txt" ,dim));
////				 p =  new BakerTransformedPointSet(new Rank1Lattice(N[i],aMult,dim));
//
//			
//			 rand = new RandomShift(noise);
//			 rqmcPts[i] = new RQMCPointSet(p, rand);
//			 }
//			 rqmcPts[0].setLabel("Lattice+Baker"+merit);
//			 listRQMC.add(rqmcPts);
//		}

//		POLYNOMIAL LATTICE
//		merits = new String[] { "P2","P4","P6", "projdept", "projdept-stardisc", "R", "projdept-resgap"
//				,	"t100" 
//				};
//		path = "/u/puchhamf/misc/latnetbuilder/output/GenzGaussianPeak/d5a2u0p5/ord/0p05k/poly/";
//		for (String merit : merits) {
//			rqmcPts = new RQMCPointSet[numSets];
//			for (i = 0; i < numSets; ++i) {
//
//				p = new DigitalNetBase2FromLatNetBuilder(path + merit + "/2^" + (mink + i) + "/outputMachine.txt",
//						mink + i, 31, dim);
//
//				rand = new RandomShift(noise);
//				rqmcPts[i] = new RQMCPointSet(p, rand);
//			}
//			rqmcPts[0].setLabel("PolyLat" + merit);
//			listRQMC.add(rqmcPts);
//		}
//
//		merits = new String[] { "P2", "P4","P6","R" };
//		for (String merit : merits) {
//			rqmcPts = new RQMCPointSet[numSets];
//			for (i = 0; i < numSets; ++i) {
//
//				p = new BakerTransformedPointSet(new DigitalNetBase2FromLatNetBuilder(
//						path + merit + "/2^" + (mink + i) + "/outputMachine.txt", mink + i, 31, dim));
//
//				rand = new RandomShift(noise);
//				rqmcPts[i] = new RQMCPointSet(p, rand);
//			}
//			rqmcPts[0].setLabel("PolyLat+baker" + merit);
//			listRQMC.add(rqmcPts);
//		}

		// SOBOL
		
		// Sobol - Lemieux
				rqmcPts = new RQMCPointSet[numSets];
				for (i = 0; i < numSets; ++i) {
					p = new SobolSequence(i + mink, 31, dim);
					rand = new RandomShift(noise);
					rqmcPts[i] = new RQMCPointSet(p, rand);
				}
				rqmcPts[0].setLabel("Sobol+Lemieux");
				listRQMC.add(rqmcPts);

				// Sobol - Kuo
				rqmcPts = new RQMCPointSet[numSets];
				for (i = 0; i < numSets; ++i) {
					p = new SobolSequence("/u/puchhamf/misc/latnetbuilder/data/new-joe-kuo-6.21201", i + mink, 31, dim);
					rand = new RandomShift(noise);
					rqmcPts[i] = new RQMCPointSet(p, rand);
				}
				rqmcPts[0].setLabel("Sobol+Kuo");
				listRQMC.add(rqmcPts);

		
		merits = new String[] { "P2","P4","P6", "R", "projdept", "projdept-stardisc", "projdept-resgap", "t100" };
		path = "/u/puchhamf/misc/latnetbuilder/output/GenzGaussianPeak/d5a2u0p5/ord/0p05k/sobol/";
		for (String merit : merits) {
			rqmcPts = new RQMCPointSet[numSets];
			for (i = 0; i < numSets; ++i) {

				p = new DigitalNetBase2FromLatNetBuilder(path + merit + "/2^" + (mink + i) + "/outputMachine.txt",
						mink + i, 31, dim);

				rand = new RandomShift(noise);
				rqmcPts[i] = new RQMCPointSet(p, rand);
			}
			rqmcPts[0].setLabel("Sobol+" + merit);
			listRQMC.add(rqmcPts);
		}

		
//		int[][] aa = {
//				{1, 3455, 1899, 2921, 3663, 2823, 3977, 2761, 255, 845, 3029, 3831, 2089, 3691, 1771, 3907, 337, 3735, 1373, 1795}, // 13
//
//				{ 1, 6915, 4877, 7479, 1203, 3941, 2159, 3225, 5219, 6307, 2643, 633, 7139, 869, 7239, 7019, 8151, 3853, 8019, 5731}, // 14
//
//				{ 1, 12033, 3801, 5023, 10647, 14127, 12751, 7461, 11901, 1167, 14349, 1951, 2209, 7397, 2505, 5675, 12195, 1801, 7707, 13443 }, // 15
//
//				{ 1, 25015, 11675, 7425, 3289, 17821, 5649, 32161, 10285, 12031, 26337, 13403, 14547, 18661, 7993, 1299, 15111, 12735, 13129, 12655 }, // 16
//
//				{ 1, 38401, 48799, 17301, 59639, 20297, 26805, 53109, 4365, 14055, 5023, 48499, 37937, 5155, 44255, 61671, 11409, 38529, 61887, 19183}, // 17
//
//				{ 1, 96407, 36479, 31333, 63411, 80945, 24597, 41083, 70179, 42983, 62013, 48035, 80011, 105415, 108151, 68869, 104973, 20719, 72257, 59193 }, // 18
//
//				{ 1, 154805, 243089, 211205, 258913, 18107, 174117, 67287, 3585, 155767, 31401, 154275, 35513, 36509, 162377, 51021, 88413, 190981, 145989, 257551}, // 19
//
//				{1, 387275, 457903, 282967, 117983, 355873, 439959, 109733, 382437, 297385, 267803, 68841, 343399, 171303, 420841, 136437, 423733, 355591, 415917, 406205} // 20
//		};
//		
//		rqmcPts = new RQMCPointSet[numSets];
//		for (i = 0; i < numSets; ++i) {
//
//			p = new Rank1Lattice(N[i],aa[i],
//					 dim);
//
//			rand = new RandomShift(noise);
//			rqmcPts[i] = new RQMCPointSet(p, rand);
//		}
//		rqmcPts[0].setLabel("Lat+shift");
//		listRQMC.add(rqmcPts);
//	
//	
//	rqmcPts = new RQMCPointSet[numSets];
//	for (i = 0; i < numSets; ++i) {
//
//		p = new BakerTransformedPointSet(new Rank1Lattice(N[i],aa[i],
//				 dim));
//
//		rand = new RandomShift(noise);
//		rqmcPts[i] = new RQMCPointSet(p, rand);
//	}
//	rqmcPts[0].setLabel("Lat+baker");
//	listRQMC.add(rqmcPts);

		
		// Stratification
//		rqmcPts = new RQMCPointSet[numSets];
//		int k;
//		for (i = 0; i < numSets; ++i) {
//			k = (int) Math.round(Math.pow(Num.TWOEXP[i + mink], 1.0 / (double) (dim)));
//			p = new StratifiedUnitCube(k, dim);
//
//			rand = new RandomShift(noise);
//			rqmcPts[i] = new RQMCPointSet(p, rand);
//		}
//		rqmcPts[0].setLabel("Stratification");
//		listRQMC.add(rqmcPts);

		// FAURE + LMS
//		rqmcPts = new RQMCPointSet[numSets];
//		for (i = 0; i < numSets; ++i) {
//
//			p = new FaureSequence(2, i + mink, 31, 31, dim);
//
//			rand = new LMScrambleShift(noise);
//			rqmcPts[i] = new RQMCPointSet(p, rand);
//		}
//		rqmcPts[0].setLabel("Faure+LMS");
//		listRQMC.add(rqmcPts);

		// Sobol + LMS
//		rqmcPts = new RQMCPointSet[numSets];
//		for (i = 0; i < numSets; ++i) {
//
//			p = new SobolSequence(i + mink, 31, dim);
//
//			rand = new LMScrambleShift(noise);
//			rqmcPts[i] = new RQMCPointSet(p, rand);
//		}
//		rqmcPts[0].setLabel("Sobol+LMS");
//		listRQMC.add(rqmcPts);

		// Sobol+NUS
//		rqmcPts = new RQMCPointSet[numSets];
//		for (i = 0; i < numSets; ++i) {
//			CachedPointSet cp = new CachedPointSet(new SobolSequence(i + mink, 31, dim));
//			cp.setRandomizeParent(false);
//			p = cp;
//
//			rand = new NestedUniformScrambling(noise);
//			rqmcPts[i] = new RQMCPointSet(p, rand);
//		}
//		rqmcPts[0].setLabel("Sobol+NUS");
//		listRQMC.add(rqmcPts);

		boolean makePgfTable = true;
		boolean printReport = true;
		boolean details = true;

		// Perform an experiment with the list of series of RQMC point sets.
		// This list contains two series: lattice and Sobol.
		ArrayList<PgfDataTable> listCurves = new ArrayList<PgfDataTable>();
		RQMCExperimentSeries experSeries = new RQMCExperimentSeries(listRQMC.get(0), basis);
		experSeries.setExecutionDisplay(details);
		file = new File(outdir + "reportRQMC.txt");
//		file.getParentFile().mkdirs();
		fw = new FileWriter(file);

		fw.write(modelDescr + experSeries.testVarianceRateManyPointTypes(model, listRQMC, m, numSkipReg, makePgfTable,
				printReport, details, listCurves));
		fw.close();

		// Produces LaTeX code to draw these curves with pgfplot.
		sb = new StringBuffer("");
		sb.append(PgfDataTable.pgfplotFileHeader());
		sb.append(PgfDataTable.drawPgfPlotManyCurves(modelDescr + ": Mean values", "axis", 3, 1, listCurves, basis, "",
				" "));
		sb.append(PgfDataTable.pgfplotEndDocument());
		file = new File(outdir + "plotMean.tex");
		fw = new FileWriter(file);
		fw.write(sb.toString());
		fw.close();

		sb = new StringBuffer("");
		sb.append(PgfDataTable.pgfplotFileHeader());
		sb.append(PgfDataTable.drawPgfPlotManyCurves(modelDescr + ": Variance", "axis", 3, 4, listCurves, basis, "",
				" "));
		sb.append(PgfDataTable.pgfplotEndDocument());
		file = new File(outdir + "plotVariance.tex");
		fw = new FileWriter(file);
		fw.write(sb.toString());
		fw.close();

	}

}
