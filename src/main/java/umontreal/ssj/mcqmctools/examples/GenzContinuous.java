package umontreal.ssj.mcqmctools.examples;

import java.util.Arrays;

import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.util.Num;

public class GenzContinuous implements MonteCarloModelDouble{

	double dim;
	double[] a;
	double[] u;
	double performance;
	
	public GenzContinuous(double[] a, double[] u,int dim) {
		this.a = new double [dim];
		this.u = new double [dim];
		for(int j =0; j < dim; ++j) {
			this.a[j] = a[j]; 
			this.u[j] = u[j]; 
		}
		this.dim = dim;
	}
	
	public GenzContinuous(double[] a,double[] u) {
		this(a,u, Math.min(a.length,u.length));
	}
	
	public GenzContinuous(double a, double u,int dim) {
		this.a = new double[dim];
		Arrays.fill(this.a,a);
		this.u = new double[dim];
		Arrays.fill(this.u,u);
		this.dim = dim;
	}
	
	@Override
	public void simulate(RandomStream stream) {
		 double exponent = 0.0;
		for(int j =0; j < dim; ++j)
			exponent += a[j]*Math.abs(stream.nextDouble() - u[j]);
		performance = Math.exp(-exponent);
	}

	@Override
	public double getPerformance() {
		return performance;
	}
	
	public String toString() {
		String str = "GenzContinuous [a = {" + a[0];
		for(int j = 1; j < dim; ++j)
			str += ", " +a[j];
		str+="}, ";
		str+= "u = {" + u[0];
				for(int j = 1; j < dim; ++j)
					str += ", " +u[j];
		str+="}]\n";
		return str;
	}

}
