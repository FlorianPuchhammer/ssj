package umontreal.ssj.mcqmctools.examples;

import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.RandomStream;

public class SumOfNormals implements MonteCarloModelDouble {

	double[] mu;
	double[] sigma;
	int dim;
	double perf;
	double norma;
	
	
	public SumOfNormals(double[] mu, double sigma[],double norma) {
		this.mu = mu;
		this.sigma = sigma;
		this.dim = mu.length;
		this.norma = norma;
	}
	
	public SumOfNormals(double[] mu, double sigma[]) {
		this(mu,sigma,1.0);
	}
	
	@Override
	public void simulate(RandomStream stream) {
		perf = 0.0;
		for(int j = 0; j< dim; ++j)
			perf+=NormalDist.inverseF(mu[j],sigma[j],stream.nextDouble());
		
		
	}

	@Override
	public double getPerformance() {
		return (perf/norma);
	}
	
	public String toString() {
		return "Sum of " + dim + " normal variables";
	}

}
