package umontreal.ssj.mcqmctools.examples;

import java.util.Arrays;

import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.rng.RandomStream;

public class GFunction implements MonteCarloModelDouble{

	double dim;
	double[] a;
	double performance;
	
	public GFunction(double[] a, int dim) {
		this.a = new double [dim];
		for(int j =0; j < dim; ++j)
			this.a[j] = a[j]; 
		this.dim = dim;
	}
	
	public GFunction(double[] a) {
		this(a,a.length);
	}
	
	public GFunction(double a, int dim) {
		this.a = new double[dim];
		Arrays.fill(this.a,a);
		this.dim = dim;
	}
	
	@Override
	public void simulate(RandomStream stream) {
		 performance = 1.0;
		for(int j =0; j < dim; ++j)
			performance*= (Math.abs(4.0*stream.nextDouble() - 2.0) + a[j])/(1.0+a[j]);
		
	}

	@Override
	public double getPerformance() {
		return performance;
	}
	
	public String toString() {
		String str = "G-Function [a = {" + a[0];
		for(int j = 1; j < dim; ++j)
			str += ", " +a[j];
		str+="}]\n";
		return str;
	}

}
