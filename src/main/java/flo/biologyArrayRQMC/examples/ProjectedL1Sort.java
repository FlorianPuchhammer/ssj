package flo.biologyArrayRQMC.examples;

import umontreal.ssj.util.sort.MultDimToOneDimSort;

public class ProjectedL1Sort extends MultDimToOneDimSort {

	double[] X0;
	double[] c;
	double t;
	double x02;
	public ProjectedL1Sort(double[] X0, double[] c, double t, double N0) {
		this.X0 = X0;
		this.c = c;
		this.t = t;
		x02 = N0;
		for(double x : X0)
			x02 -= x;
	}
	@Override
	public double scoreFunction(double[] v) {		
	return( v[0] - X0[0]- 0.5 * c[0] * X0[0]*(X0[0]-1.0) * X0[1]*t + c[1]* t /6.0*X0[0]*(X0[0]-1.0)*(X0[0]-2.0)-c[2]*t*x02+c[3]*t*X0[0]);
	}

}