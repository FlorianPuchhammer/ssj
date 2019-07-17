package umontreal.ssj.mcqmctools.examples;

import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.RandomStream;

public class MultiNormalIndependent implements MonteCarloModelDoubleArray {
	int dim;
	double [] mus;
	double [] sigmas;
	double[] performance;
	
	public MultiNormalIndependent(double[] mus, double[] sigmas) {
		this.mus = mus;
		this.sigmas = sigmas;
		dim = this.mus.length;
		performance = new double[dim];
		
	}
	@Override
	public void simulate(RandomStream stream) {
		for(int j = 0; j < dim; j++) {
			performance[j] = NormalDist.inverseF(mus[j],sigmas[j],stream.nextDouble());
		}
	}

	@Override
	public double[] getPerformance() {
		return performance;
	}

	@Override
	public int getPerformanceDim() {
		return dim;
	}
	
	@Override
	public String toString() {
		return "MultiNormalInd";
	}
}
