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
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.hups.SortedAndCutPointSet;
import umontreal.ssj.hups.StratifiedUnitCube;
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
		String outdir = "/u/puchhamf/misc/jars/creditMetrics/data/creditMetrics/KP6/";
		FileWriter fw;
		File file;
		StringBuffer sb = new StringBuffer("");

		String filename = "KP.dat";
		model = new CreditMetrics(outdir + filename, noise);
		double norma = ((CreditMetrics) model).nom();
		((CreditMetrics) model).normalize(norma);
		modelDescr = model.toString();
		dim = ((CreditMetrics) model).getDimension();
		
		sb.append( ((CreditMetrics) model).toStringHeader());
		String str = sb.toString();
		System.out.println(str);

		Tally statValue = new Tally("RQMC stat for" + modelDescr);

		// Define the RQMC point sets to be used in experiments.
		int basis = 2; // Basis for the loglog plots.
		int numSkipReg = 0; // Number of sets skipped for the regression.
		int mink = 9; // first log(N) considered
		int i;
		int m = 100; // Number of RQMC randomizations.
//		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152 }; // 13
		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,262144, 524288, 1048576};
		int numSets = N.length; // Number of sets in the series.

		// Create a list of series of RQMC point sets.
		ArrayList<RQMCPointSet[]> listRQMC = new ArrayList<RQMCPointSet[]>();
		PointSet p;
		PointSetRandomization rand;
		RQMCPointSet[] rqmcPts;

		// Independent points (Monte Carlo)
		rqmcPts = new RQMCPointSet[numSets];
		for (i = 0; i < numSets; ++i) {
			p = new IndependentPointsCached(N[i], dim);
			rand = new RandomShift(noise);
			rqmcPts[i] = new RQMCPointSet(p, rand);
		}
		rqmcPts[0].setLabel("Independent points");
		listRQMC.add(rqmcPts);

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
		rqmcPts = new RQMCPointSet[numSets];
		for (i = 0; i < numSets; ++i) {

			p = new FaureSequence(2, i + mink, 31, 31, dim);

			rand = new LMScrambleShift(noise);
			rqmcPts[i] = new RQMCPointSet(p, rand);
		}
		rqmcPts[0].setLabel("Faure+LMS");
		listRQMC.add(rqmcPts);
				
			

		
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

		fw.write(str + experSeries.testVarianceRateManyPointTypes(model, listRQMC, m, numSkipReg, makePgfTable, printReport,
				details, listCurves));
		fw.close();

		// Produces LaTeX code to draw these curves with pgfplot.
		sb = new StringBuffer("");
		sb.append(PgfDataTable.pgfplotFileHeader ());
		sb.append(PgfDataTable.drawPgfPlotManyCurves(model.toString() + ": Mean values", "axis", 3, 1, listCurves, basis, "",
				" "));
		sb.append(PgfDataTable.pgfplotEndDocument());
		file = new File(outdir + "plotMean.tex");
		fw = new FileWriter(file);
		fw.write(sb.toString());
		fw.close();
		
		
		sb = new StringBuffer("");
		sb.append(PgfDataTable.pgfplotFileHeader ());
		sb.append(PgfDataTable.drawPgfPlotManyCurves(model.toString() + ": Variance", "axis", 3, 4, listCurves, basis, "",
				" "));
		sb.append(PgfDataTable.pgfplotEndDocument());
		file = new File(outdir + "plotVariance.tex");
		fw = new FileWriter(file);
		fw.write(sb.toString());
		fw.close();

	}

}
