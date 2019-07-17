package umontreal.ssj.mcqmctools.examples;

import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.probdist.LognormalDist;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.RandomStream;

public class BucklingStrengthVars implements MonteCarloModelDoubleArray {
	
	/**
	 * Order of parameters: b, t, sigma0, E, delta0, eta
	 */
	double [] mu;
	double [] sigma;
	double [] performance; // 2t/b, lambda , delta0, eta
	
	public BucklingStrengthVars(double[]mu,double[] sigma) {
		this.mu = mu;
		this.sigma = sigma;
		performance = new double[4];
	}
	
	private double transformMu(double mu, double sigma) {
		return ( Math.log(mu) - 0.5 * Math.log(sigma*sigma /( mu*mu) + 1.0));
	}
	
	private double transformSigma(double mu, double sigma) {
		return (   Math.sqrt( Math.log(1.0 + sigma*sigma/(mu * mu))));
	}
	
	/**
	 * b,t,delta0,eta,lambda
	 */
	@Override
	public void simulate(RandomStream stream) {
		// 2t/b
		performance[0] = 2.0 * Math.exp(NormalDist.inverseF(transformMu(mu[1],sigma[1]),transformSigma(mu[1],sigma[1]) ,stream.nextDouble())) / NormalDist.inverseF(mu[0],sigma[0],stream.nextDouble()); //b
		// lambda
		performance[1] = 2.0 * Math.sqrt( Math.exp(NormalDist.inverseF(transformMu(mu[2],sigma[2]),transformSigma(mu[2],sigma[2]),stream.nextDouble())) / NormalDist.inverseF(mu[3],sigma[3],stream.nextDouble()) )/ performance[0];
		//delta0
		performance[2] = NormalDist.inverseF(mu[4],sigma[4],stream.nextDouble());
		//eta
		performance[3] = NormalDist.inverseF(mu[5],sigma[5],stream.nextDouble());//eta
	}

	@Override
	public double[] getPerformance() {
		
		return performance;
	}

	@Override
	public int getPerformanceDim() {
		return performance.length;
	}
	
	

}
