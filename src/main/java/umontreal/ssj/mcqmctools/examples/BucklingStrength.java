package umontreal.ssj.mcqmctools.examples;

import umontreal.ssj.hups.DigitalNetBase2;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;

public class BucklingStrength implements MonteCarloModelDouble {

	/**
	 * Order of parameters: b, t, sigma0, E, delta0, eta
	 */
	double [] mu;
	double [] sigma;
	double  performance; 
	
	public BucklingStrength(double[]mu,double[] sigma) {
		this.mu = mu;
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
		double b = NormalDist.inverseF(mu[0],sigma[0],stream.nextDouble()); 
		double t = Math.exp(NormalDist.inverseF(transformMu(mu[1],sigma[1]),transformSigma(mu[1],sigma[1]) ,stream.nextDouble()));
		double lambda = b/t * Math.sqrt(  Math.exp(NormalDist.inverseF(transformMu(mu[2],sigma[2]),transformSigma(mu[2],sigma[2]) ,stream.nextDouble())) / NormalDist.inverseF(mu[3],sigma[3],stream.nextDouble()));
		
		performance = (2.1/lambda - 0.9/(lambda * lambda)) * (1.0 - 0.75 * NormalDist.inverseF(mu[4],sigma[4],stream.nextDouble()) / lambda) * (1.0 - 2.0 * t * NormalDist.inverseF(mu[5],sigma[5],stream.nextDouble())/b);

	}

	@Override
	public double getPerformance() {
		return performance;
	}

	public String toString() {
		return "BucklingStrength";
	}
	
	public static void main(String[] args) {
//		 int N = 32;
//		 int logN = 5;
		int N = 4194304 ;/// 4;
		int logN = 22 ;//-2;
		int m = 1;

		double[] mus = { 0.992 * 24.0, 1.05 * 0.5, 1.3 * 34.0, 0.987 * 29.0E3,0.35, 5.25};
		double[] covs = { 0.028, 0.044, 0.1235, 0.076,0.05,0.07 };
		int dim = mus.length;
		double[] sigmas = new double[dim];
		for(int j = 0; j < dim; j++)
			sigmas[j] = mus[j] * covs[j];
		MonteCarloModelDouble model = new BucklingStrength(mus, sigmas);


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
			if (dat >= 0.5169&& dat <= 0.6511)
				++hits;
		}
		
		System.out.println("Relative mass of the interval: = " + (double) hits / (double) N);

	}
}
