package flo;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.BucklingStrengthVars;
import umontreal.ssj.mcqmctools.examples.MultiNormalIndependent;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.density.CDEBucklingStrength;
import umontreal.ssj.stat.density.CDECantilever;
import umontreal.ssj.stat.density.ConditionalDensityEstimator;
import umontreal.ssj.stat.density.DensityEstimator;
import umontreal.ssj.stat.density.LLBucklingStrength;
import umontreal.ssj.stat.density.LLCantilever;
import umontreal.ssj.stat.list.ListOfTallies;

public class IVvsCoeff {
	private static double[] genEvalPoints(int numPts, double a, double b, RandomStream stream) {
		double[] evalPts = new double[numPts];
		double invNumPts = 1.0 / ((double) numPts);
		for (int i = 0; i < numPts; i++)
			evalPts[i] = a + (b - a) * ((double) i + stream.nextDouble()) * invNumPts;
		return evalPts;
	}

	private static double[] genEvalPoints(int numPts, double a, double b) {
		double[] evalPts = new double[numPts];
		double invNumPts = 1.0 / ((double) numPts);
		for (int i = 0; i < numPts; i++)
			evalPts[i] = a + (b - a) * ((double) i + 0.5) * invNumPts;
		return evalPts;
	}

	public static void main(String[] args) throws IOException {
		int m = 100;
		int N = 512  *  512 ; // * 256;
		int mink = 9 +9 ;// + 8;
		int numEvalPts = 128;
		int numP = 64;

		// CANTI
		// double[] mus = { 2.9E7, 500.0, 1000.0 };
		// double[] sigmas = { 1.45E6, 100.0, 100.0 };
		// MonteCarloModelDoubleArray model = new MultiNormalIndependent(mus, sigmas);
		// int dim = 3;

		// CANTI
		// double a = 0.407;
		// double b = 1.515;
		//
		// double D0 = 2.2535;
		// a = (a + 1) * D0;
		// b = (b + 1) * D0;

		// BUCKLING
		double a = 0.5169;
		double b = 0.6511;
		double[] mus = { 0.992 * 24.0, 1.05 * 0.5, 1.3 * 34.0, 0.987 * 29.0E3, 0.35, 5.25 };
		double[] covs = { 0.028, 0.044, 0.1235, 0.076, 0.05, 0.07 };
		int dim = mus.length;
		double[] sigmas = new double[dim];
		for (int j = 0; j < dim; j++)
			sigmas[j] = mus[j] * covs[j];
		MonteCarloModelDoubleArray model = new BucklingStrengthVars(mus, sigmas);

		RandomStream noise = new MRG32k3a();
		double[] evalPts = genEvalPoints(numEvalPts, a, b, noise);
		 double[] pp = genEvalPoints(numP,-0.5,1.5);
//		double[] pp = genEvalPoints(numP, -0.015625, 0.046875);// CDEBuckling
//		 double[] pp = genEvalPoints(numP,0.12,0.17);// LLBuckling
		// pp[0] = 0.0;

		 double L = 100.0;
		 double t = 2.0;
		 double w = 4.0;

		double[][][] data = new double[m][N][model.getPerformanceDim()];
		PointSet p = new SobolSequence(mink, 31, dim);
		PointSetRandomization rand = new LMScrambleShift(noise);
		RQMCPointSet rqmc = new RQMCPointSet(p, rand);
		ListOfTallies<Tally> statRepsList = ListOfTallies.createWithTally(model.getPerformanceDim());

		RQMCExperiment.simulReplicatesRQMC(model, rqmc, m, statRepsList, data);

		double[][] density = new double[m][evalPts.length];

		System.out.println("Sufficient Memory!");

		// ConditionalDensityEstimator cde = new LLCantilever(L, t, w, mus[0],
		// sigmas[0], mus[1], sigmas[1], mus[2],
		// sigmas[2], pp[0]);
		ConditionalDensityEstimator cde = new LLBucklingStrength(mus[4],sigmas[4],mus[5],sigmas[5],pp[0]);
//		ConditionalDensityEstimator cde = new CDEBucklingStrength(mus[4], sigmas[4], mus[5], sigmas[5], pp[0]);

		double[] iv = new double[pp.length];
		double[] var = new double[evalPts.length];
		for (int we = 0; we < pp.length; we++) {
			((LLBucklingStrength) cde).setP(pp[we]);
			for (int rep = 0; rep < m; rep++) {
				density[rep] = cde.evalDensity(evalPts, data[rep]);
			}

			// iv[we] = Math.log(DensityEstimator.computeIV(density,a,b,var))/Math.log(2.0);
			iv[we] = DensityEstimator.computeIV(density, a, b, var);

			// for (double v : iv) {
			// System.out.println(v);
			// }
		}

		String[] axisTitle = { "p", "IV" };
		FileWriter fw = new FileWriter("LLBuckling_IV_vs_p.tex");
		fw.write(DensityEstimator.plotDensity(pp, iv, "LLBuckling: IV vs p", axisTitle));
		fw.close();

		double[] min = { 0, 1000 };
		for (int i = 0; i < pp.length; i++) {
			if (min[1] > iv[i]) {
				min[1] = iv[i];
				min[0] = pp[i];
			}
		}

		System.out.println("Min( IV)=\t" + min[1]);
		System.out.println("At p =\t" + min[0]);
		System.out.println("A -- O K ! ! !");

	}

}
