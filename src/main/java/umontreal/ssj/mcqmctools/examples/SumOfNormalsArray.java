package umontreal.ssj.mcqmctools.examples;

import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.RandomStream;

public class SumOfNormalsArray implements MonteCarloModelDoubleArray {

	double[] mu;
	double[] sigma;
	int dim;
	double[] perf ;
	
	public SumOfNormalsArray(double[] mu, double sigma[]) {
		this.mu = mu;
		this.sigma = sigma;
		this.dim = mu.length;
		perf = new double[1];
	}
	
	
	
	public void simulate(RandomStream stream) {
		perf[0] = 0.0;
		for(int j = 0; j< dim; ++j)
			perf[0]+=NormalDist.inverseF(mu[j],sigma[j],stream.nextDouble());
	}

	@Override
	public double[] getPerformance() {
	return perf;
	}

	@Override
	public int getPerformanceDim() {
		return 1;
	}

}
