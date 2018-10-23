package umontreal.ssj.mcqmctools.examples;

import umontreal.ssj.mcqmctools.MonteCarloModelDensityKnown;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.RandomStream;

public class SumOfStandardNormalsNormalized implements MonteCarloModelDensityKnown {

	int dimension;
	double sum;
	
	public SumOfStandardNormalsNormalized(int dimension) {
		this.dimension = dimension;
	}
	@Override
	public void simulate(RandomStream stream) {
		sum = 0.0;
		for(int j = 0; j < dimension; j++) 
			sum += NormalDist.inverseF01(stream.nextDouble());
		sum /= Math.sqrt(dimension);
		

	}

	@Override
	public double getPerformance() {
		return sum;
	}

	@Override
	public double density(double x) {
		return NormalDist.density01(x);
	}

	@Override
	public double cdf(double x) {
		return NormalDist.cdf01(x);
	}
	
	public String toString() {
		return "Normalized sum of " + dimension + "standard normals";
	}

}
