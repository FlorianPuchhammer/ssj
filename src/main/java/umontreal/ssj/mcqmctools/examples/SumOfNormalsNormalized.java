package umontreal.ssj.mcqmctools.examples;

import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.RandomStream;

public class SumOfNormalsNormalized implements MonteCarloModelDouble {

	double[] a;
	double sum;
	double stdDev;
	
	int dim;
	
	public SumOfNormalsNormalized(double[] a) {
		this.a = a;
		stdDev = computeStdDev();
		dim = a.length;
	}
	
	private double computeStdDev() {
		double sigma = 0.0;
		for(double s:a)
			sigma+=s*s;
		return Math.sqrt(sigma);
	}
	
	@Override
	public void simulate(RandomStream stream) {
		sum = 0.0;
		for(int j = 0; j < dim; j++)
			sum += a[j] * NormalDist.inverseF01(stream.nextDouble());
	}

	@Override
	public double getPerformance() {
		return (sum/stdDev);
	}
	
	public String toString() {
		return "Normalized sum of " + dim + " normals";
	}

	

}
