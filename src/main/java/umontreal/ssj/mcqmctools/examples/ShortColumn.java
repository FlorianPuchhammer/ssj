package umontreal.ssj.mcqmctools.examples;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.CholeskyDecomposition;
import umontreal.ssj.hups.DigitalNetBase2;
import umontreal.ssj.hups.EmptyRandomization;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.randvarmulti.MultinormalCholeskyGen;
import umontreal.ssj.randvarmulti.MultinormalPCAGen;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;

public class ShortColumn implements MonteCarloModelDouble {

	double h;
	double b;
	double muY, muM, muP;
	double sigmaY;
	DoubleMatrix2D trafoMat;
	double  performance;
	double[][] sigma;
	
	/**
	 * 
	 * @param h
	 * @param b
	 * @param muY
	 * @param muM
	 * @param muP
	 * @param sigmaY
	 * @param sigma
	 */
	public ShortColumn(double h, double b, double muY, double muM, double muP, double sigmaY, DoubleMatrix2D sigma) {
		this.h = h;
		this.b = b;
		this.muY = muY;
		this.muM = muM;
		this.muP = muP;
		this.sigmaY = sigmaY;
		this.trafoMat = MultinormalPCAGen.decompPCA(sigma);
//		this.trafoMat = new CholeskyDecomposition(sigma));
//		this.trafoMat = DoubleMatrix2D.identity(2);
		
		
	}
	/**
	 * 
	 * @param h
	 * @param b
	 * @param muY
	 * @param muM
	 * @param muP
	 * @param sigmaY
	 * @param sigma
	 */
	public ShortColumn(double h, double b, double muY, double muM, double muP, double sigmaY, double[][] sigma) {
		this.h = h;
		this.b = b;
		this.muY = transformMu(muY,sigmaY);
		this.muM = muM;
		this.muP = muP;
		this.sigmaY = transformSigma(muY,sigmaY);
		this.trafoMat = MultinormalPCAGen.decompPCA(sigma);
//		this.trafoMat = DoubleFactory2D.dense.make(sigma);
//		System.out.println(trafoMat);
		this.sigma = sigma;
	}
	
	private double transformMu(double mu, double sigma) {
		return ( Math.log(mu) - 0.5 * Math.log(sigma*sigma /( mu*mu) + 1.0));
	}
	
	private double transformSigma(double mu, double sigma) {
		return (   Math.sqrt( Math.log(1.0 + sigma*sigma/(mu * mu))));
	}
	
	@Override
	public void simulate(RandomStream stream) {
		double Y = Math.exp(NormalDist.inverseF(muY,sigmaY,stream.nextDouble()));
		double [] u = new double[2];
		double [] z = new double[2];
		u[0] = NormalDist.inverseF01(stream.nextDouble());
		u[1] = NormalDist.inverseF01(stream.nextDouble());
		for (int j = 0; j < 2; j++) {
			z[j] = 0;
			for (int c = 0; c < 2; c++)
//				z[j] += trafoMat.getQuick(j, c) * u[c];
				z[j] += sigma[j][c]*u[c];
		}
		z[0] += muM;
		z[1] += muP;
		double fac = 1.0/(b * h * h * Y);
//		System.out.println("(" + z[0] + ", " + z[1] + ")");
		performance = 1.0 -  4.0 * z[0] * fac - z[1] * z[1] * fac / (b * Y);
	}

	@Override
	public double getPerformance() {
		return performance;
	}
	
	public static void main(String[] args) {
//		 int N = 32;
//		 int logN = 5;
		int N = 4194304 ;// / 4;
		int logN = 22 ;// -2;
		int m = 1;
		int dim;

		double h = 15.0;
		double b = 5.0;
		double muY = 5.0;
		double muM = 2000.0;
		double muP = 500.0;
		double sigmaY = 0.5;
//		double[][] sigma = {{400.0, 100.0}, {100.0, 100.0}};
//		double [][] sigma = {{160000.0, 20000.0}, {20000.0, 10000.0}};
		double[][] sigma = {{400.0, 50.0}, {0.0, 86.60254037844386}};
//		double [][] sigma = {{400.0, 50.0}, {0.0, 50.0 * Math.sqrt(3.0)}};
		MonteCarloModelDouble model = new ShortColumn(h,b,muY,muM,muP,sigmaY,sigma);

		dim = 3;

		DigitalNetBase2 pts = (new SobolSequence(logN, 31, dim));

		double[][] data = new double[m][];
		Tally statReps = new Tally("Stats on RQMCExperiment");
		PointSetRandomization rand = new LMScrambleShift(new MRG32k3a());
		RQMCPointSet sobolNet = new RQMCPointSet(pts, rand);

		RQMCExperiment.simulReplicatesRQMC(model, sobolNet, m, statReps, data);

		double[] percentiles = { 0.49, 0.5, 0.51, 2.49, 2.50, 2.51, 97.49, 97.50, 97.51, 99.49, 99.5, 99.51 };
		for (double perc : percentiles)
			System.out.println(perc + "%-percentile: " + data[m - 1][(int) Math.ceil((perc * N) / 100) - 1]);
		System.out.println(data[m - 1][N - 2]);

		int hits = 0;
		for (double dat : data[m - 1]) {
			if (dat >= -5.338 && dat <= -0.528)
				++hits;
		}
		
		System.out.println("Relative mass of the interval: = " + (double) hits / (double) N);

	}

}
