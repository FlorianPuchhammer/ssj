package umontreal.ssj.mcqmctools.examples;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import umontreal.ssj.hups.BakerTransformedPointSet;
import umontreal.ssj.hups.IndependentPointsCached;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.RandomShift;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.latnetbuilder.DigitalNetBase2FromLatNetBuilder;
import umontreal.ssj.latnetbuilder.Rank1LatticeFromLatNetBuilder;
import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.mcqmctools.QMCExperimentSeries;
import umontreal.ssj.mcqmctools.RQMCExperimentSeries;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.PgfDataTable;
import umontreal.ssj.util.Num;

public class testQMCExperimentDouble {
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



//		double a = 0.5;
//		dim = 5;
//		model = new GFunction(a, dim);
//		sb.append(model.toString());
//		modelDescr = "GFunction";
//		System.out.println(sb.toString());
		
		double a = 1.0;
		dim = 5;
		double u = 0.5;
		model = new GenzGaussianPeak(a, u,dim);
		double trueMean = 1.0;
		double aa,uu;
		for(int j=0; j < dim; ++j) {
			aa = ((GenzGaussianPeak) model).a[j];
			uu = ((GenzGaussianPeak) model).u[j];
			trueMean *= 0.5 * Math.sqrt(Math.PI) * (Num.erf(aa * uu ) + Num.erf(aa * (1.0 - uu))) / aa;
		}
		modelDescr = "GenzGaussianPeak";
		
//		double a = 1.0;
//		dim = 5;
//		double u = 0.5;
//		model = new GenzContinuous(a, u,dim);
//		double trueMean = 1.0;
//		double aa,uu;
//		for(int j=0; j < dim; ++j) {
//			aa = ((GenzContinuous) model).a[j];
//			uu = ((GenzContinuous) model).u[j];
//			trueMean *= 2.0 * (1.0 - Math.exp(-0.5*a))/a;
//		}
//		modelDescr = "GenzContinuous";
		
		sb.append(model.toString());
		
		System.out.println(sb.toString());


		// Define the RQMC point sets to be used in experiments.
		int basis = 2; // Basis for the loglog plots.
		int numSkipReg = 0; // Number of sets skipped for the regression.
		int mink = 13; // first log(N) considered
		int i;
//		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152 }; // 13
		int[] N = { 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 };
		int numSets = N.length; // Number of sets in the series.

		// Create a list of series of RQMC point sets.
		ArrayList<PointSet[]> listQMC = new ArrayList<PointSet[]>();
		PointSet p;
		PointSet[] qmcPts;
		String[] merits;
		String path;

		// Independent points (Monte Carlo)
//		qmcPts = new PointSet[numSets];
//		for (i = 0; i < numSets; ++i) {
//			p = new IndependentPointsCached(N[i], dim);
//			p.randomize(noise);
//			qmcPts[i] = p;
//		}
//		listQMC.add(qmcPts);

////		ORDINARY LATTICE
////		 merits = new String[] {"P2","R0p2","R1","R2","spectral"};
//		 merits = new String[] {"P2","P4","P6","R1","R2","R4","R6","spectral"};
////		 merits = new String[] {"P2", "R1"};
//
//		 path = "/u/puchhamf/misc/latnetbuilder/output/GenzGaussianPeak/d5a1u0p5/ord/sobFlo/rank1/";
//		
//		//Lat+Shift
//		for(String merit : merits) {
//			 qmcPts = new PointSet[numSets];
//			 for (i = 0; i < numSets; ++i) {
//			
//			 p = new Rank1LatticeFromLatNetBuilder(path + merit + "/2^" +(mink+i) + "/outputMachine.txt" ,dim);
//			
//			 qmcPts[i] = p;
//			 }
//			 listQMC.add(qmcPts);
//		}
//		
//		//Lat+Baker
//		for(String merit : merits) {
//			 qmcPts = new PointSet[numSets];
//			 for (i = 0; i < numSets; ++i) {
//			
//			 p =  new BakerTransformedPointSet(new Rank1LatticeFromLatNetBuilder(path + merit + "/2^" +(mink+i) + "/outputMachine.txt" ,dim));
////				 p =  new BakerTransformedPointSet(new Rank1Lattice(N[i],aMult,dim));
//
//			
//			 qmcPts[i] =p;
//			 }
//			 listQMC.add(qmcPts);
//		}

////		POLYNOMIAL LATTICE
////		merits = new String[] { "P2","P4","P6", "projdept", "projdept-stardisc", "R", "projdept-resgap"
////				,	"t100" 
////				};
//		merits = new String[] { "t100"
//				};
//		path = "/u/puchhamf/misc/latnetbuilder/output/GenzGaussianPeak/d5a2u0p5/ord/unanchoredSobolev/poly/";
//		for (String merit : merits) {
//			qmcPts = new PointSet[numSets];
//			for (i = 0; i < numSets; ++i) {
//
//				p = new DigitalNetBase2FromLatNetBuilder(path + merit + "/2^" + (mink + i) + "/outputMachine.txt",
//						mink + i, 31, dim);
//
//				qmcPts[i] = p;
//			}
//			listQMC.add(qmcPts);
//		}
//
//		merits = new String[] { "P2", "P4","P6","R" };
//		for (String merit : merits) {
//			qmcPts = new PointSet[numSets];
//			for (i = 0; i < numSets; ++i) {
//
//				p = new BakerTransformedPointSet(new DigitalNetBase2FromLatNetBuilder(
//						path + merit + "/2^" + (mink + i) + "/outputMachine.txt", mink + i, 31, dim));
//
//				qmcPts[i] = p;
//			}
//			listQMC.add(qmcPts);
//		}

		// SOBOL
		
		// Sobol - Lemieux
				qmcPts = new PointSet[numSets];
				for (i = 0; i < numSets; ++i) {
					p = new SobolSequence(i + mink, 31, dim);
					qmcPts[i] = p;
				}
				listQMC.add(qmcPts);

				// Sobol - Kuo
				qmcPts = new PointSet[numSets];
				for (i = 0; i < numSets; ++i) {
					p = new SobolSequence("/u/puchhamf/misc/latnetbuilder/data/new-joe-kuo-6.21201", i + mink, 31, dim);
					qmcPts[i] = p;
				}
				listQMC.add(qmcPts);

		
//		merits = new String[] { "P2","P4","P6", "R", "projdept", "projdept-stardisc", "projdept-resgap", "t100" };
				merits= new String[] {"t500"};
		path = "/u/puchhamf/misc/latnetbuilder/output/GenzGaussianPeak/d5a2u0p5/ord/0p05k/sob/";
		for (String merit : merits) {
			qmcPts = new PointSet[numSets];
			for (i = 0; i < numSets; ++i) {

				p = new DigitalNetBase2FromLatNetBuilder(path + merit + "/2^" + (mink + i) + "/outputMachine.txt",
						mink + i, 31, dim);

				qmcPts[i] = p;
			}
			listQMC.add(qmcPts);
		}

		


		boolean makePgfTable = true;
		boolean printReport = true;
		boolean details = true;
outdir = "AAA";
		// Perform an experiment with the list of series of RQMC point sets.
		// This list contains two series: lattice and Sobol.
		ArrayList<PgfDataTable> listCurves = new ArrayList<PgfDataTable>();
		QMCExperimentSeries experSeries = new QMCExperimentSeries(listQMC.get(0), basis,trueMean);
		experSeries.setExecutionDisplay(details);
		file = new File(outdir + "reportQMC.txt");
//		file.getParentFile().mkdirs();
		fw = new FileWriter(file);

		fw.write(modelDescr + experSeries.testErrorRateManyPointTypes(model, listQMC, numSkipReg, makePgfTable,
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
		sb.append(PgfDataTable.drawPgfPlotManyCurves(modelDescr + ": Error", "axis", 3, 4, listCurves, basis, "",
				" "));
		sb.append(PgfDataTable.pgfplotEndDocument());
		file = new File(outdir + "plotError.tex");
		fw = new FileWriter(file);
		fw.write(sb.toString());
		fw.close();

	}

}
