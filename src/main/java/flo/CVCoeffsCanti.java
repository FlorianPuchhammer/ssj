package flo;

import java.util.Arrays;

import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.MultiNormalIndependent;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.density.CDECantilever;
import umontreal.ssj.stat.density.ConditionalDensityEstimator;
import umontreal.ssj.stat.density.DensityEstimator;
import umontreal.ssj.stat.list.ListOfTallies;

public class CVCoeffsCanti {
	private static double[] genEvalPoints(int numPts, double a, double b, RandomStream stream) {
		double[] evalPts = new double[numPts];
		double invNumPts = 1.0 / ((double) numPts);
		for (int i = 0; i < numPts; i++)
			evalPts[i] = a + (b - a) * ((double) i + stream.nextDouble()) * invNumPts;
		return evalPts;
	}

	private static double denomC(double[] var, double[] coVar) {
		return (coVar[0] * coVar[0] + coVar[1] * coVar[1] + coVar[2] * coVar[2] + 2.0 * coVar[1] * var[0]
				- var[0] * var[1] + 2.0 * coVar[2] * (var[1] - coVar[1])
				+ 2.0 * coVar[0] * (var[2] - coVar[2] - coVar[1]) - var[2] * (var[0] +var[1]));
	}

	private static double c1(double[] var, double[] coVar) {
		return (coVar[1] * coVar[1] + coVar[2] * (var[1] - coVar[1]) - var[1] * var[2] + coVar[0] * (var[2] - coVar[1]))
				/ denomC(var, coVar);
	}
	
	private static double c2(double[] iv, double[] icov) {
		return( icov[2] * icov[2] - icov[2] * icov[1] + iv[0] * (icov[1] - iv[2]) + icov[0] * (iv[2]-icov[2])) / denomC(iv, icov);
	}
	
	private static double c3(double[] iv, double[] icov) {
		return ( icov[0] * icov[0] - icov[0] * (icov[2] + icov[1]) + icov[1] * iv[0] + iv[1] * (icov[2] - iv[0])) / denomC(iv, icov);
	}

	public static void main(String[] args) {
		int m = 100;
		int N = 512*512;
		int mink = 9+9;
		int numEvalPts = 128;
		
//		int m = 10;
//		int N =512*4;
//		int mink = 11;
//		int numEvalPts = 32;

		double[] mus = { 2.9E7, 500.0, 1000.0 };
		double[] sigmas = { 1.45E6, 100.0, 100.0 };
		MonteCarloModelDoubleArray model = new MultiNormalIndependent(mus, sigmas);
		int dim = 3;

		

		double a = 0.407;
		double b = 1.515;

		double D0 = 2.2535;
		a = (a + 1) * D0;
		b = (b + 1) * D0;

		RandomStream noise = new MRG32k3a();
		double[] evalPts = genEvalPoints(numEvalPts, a, b, noise);

		double L = 100.0;
		double t = 2.0;
		double w = 4.0;
		double[][] weights = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } };

		double[][][] data = new double[m][N][model.getPerformanceDim()];
		PointSet p = new SobolSequence(mink, 31, dim);
		PointSetRandomization rand = new LMScrambleShift(noise);
		RQMCPointSet rqmc = new RQMCPointSet(p, rand);
		ListOfTallies<Tally> statRepsList = ListOfTallies.createWithTally(model.getPerformanceDim());

		RQMCExperiment.simulReplicatesRQMC(model, rqmc, m, statRepsList, data);

		double[][][] density = new double[3][m][evalPts.length];

		ConditionalDensityEstimator cde = new CDECantilever(L, t, w, mus[0], sigmas[0], mus[1], sigmas[1], mus[2],
				sigmas[2], weights[0]);

		for (int we = 0; we < 3; we++) {
			((CDECantilever) cde).setWeights(weights[we]);
			for (int rep = 0; rep < m; rep++) {
				density[we][rep] = cde.evalDensity(evalPts, data[rep]);
			}
		}

		double[][] mean = new double[3][evalPts.length];

		double[][] var = new double[3][evalPts.length];
		double[][] coVar = new double[3][evalPts.length];

		for (int k = 0; k < evalPts.length; k++)
			for (int we = 0; we < 3; we++) {
				for (int r = 0; r < m; r++)
					mean[we][k] += density[we][r][k];
				mean[we][k] /= (double) m;

				for (int r = 0; r < m; r++)// 0 = (0,1), 1 = (1,2), 2 = (0,2)
					coVar[we][k] += (density[we][r][k] - mean[we][k])
							* (density[(we + 1) % 3][r][k] - mean[(we + 1) % 3][k]);
				coVar[we][k] /= (double) (m - 1);
			}

		for (int we = 0; we < 3; we++)
			var[we] = DensityEstimator.computeVariance(density[we]);

		double [] iv = new double[3];
		double [] icov = new double[3];
		double fac = (b-a)/ (double)evalPts.length;
		for (int we = 0; we < 3; we++) {
			for (int k = 0; k < evalPts.length; k++) {
				iv[we] += var[we][k];
				icov[we] += coVar[we][k];
			}
			iv[we] *= fac;
			icov[we] *= fac;
		}

		double C1 = c1(iv,icov);
		double C2 = c2(iv,icov);
		double C3 = c3(iv,icov);
		
			System.out.println("c1 =\t" + C1 + "\tc2 = \t" + C2+ "\tc3 = \t" + C3);

	}

}
