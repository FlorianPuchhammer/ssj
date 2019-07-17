package umontreal.ssj.mcqmctools.examples;

import cern.colt.matrix.DoubleMatrix2D;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.randvarmulti.MultinormalPCAGen;
import umontreal.ssj.rng.RandomStream;

public class ShortColumnVars implements MonteCarloModelDoubleArray {

	double muY, muM, muP;
	double sigmaY;
	DoubleMatrix2D trafoMat;
	double[] performance = new double[3];
	double[][] sigma;
	/**
	 * 
	 * @param muY
	 * @param muM
	 * @param muP
	 * @param sigmaY
	 * @param sigma
	 */
	public ShortColumnVars(double h, double b, double muY, double muM, double muP, double sigmaY, DoubleMatrix2D sigma) {

		this.muY = transformMu(muY,sigmaY);
		this.muM = muM;
		this.muP = muP;
		this.sigmaY = transformSigma(muY,sigmaY);
		this.trafoMat = MultinormalPCAGen.decompPCA(sigma);
	}
	/**
	 * 
	 * @param muY
	 * @param muM
	 * @param muP
	 * @param sigmaY
	 * @param sigma
	 */
	public ShortColumnVars(double muY, double muM, double muP, double sigmaY, double[][] sigma) {
		this.muY = transformMu(muY,sigmaY);
		this.muM = muM;
		this.muP = muP;
		this.sigmaY = transformSigma(muY,sigmaY);
		this.trafoMat = MultinormalPCAGen.decompPCA(sigma);
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
		performance[0] = Math.exp(NormalDist.inverseF(muY,sigmaY,stream.nextDouble()));
		double [] u = new double[2];
		u[0] = NormalDist.inverseF01(stream.nextDouble());
		u[1] = NormalDist.inverseF01(stream.nextDouble());
		for (int j = 0; j < 2; j++) {
			performance[j+1] = 0.0;
			for (int c = 0; c < 2; c++)
//				performance[j+1] += trafoMat.getQuick(j, c) * u[c];
				performance[j+1] += sigma[j][ c] * u[c];

		}
		performance[1] += muM;
		performance[2] += muP;

	}

	@Override
	public double[] getPerformance() {
		return performance;
	}

	@Override
	public int getPerformanceDim() {
		return performance.length;
	}
	
	@Override
	public String toString() {
		return "ShortColumn";
	}

}
