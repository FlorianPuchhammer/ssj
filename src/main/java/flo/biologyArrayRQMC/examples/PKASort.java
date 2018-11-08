package flo.biologyArrayRQMC.examples;

import umontreal.ssj.util.sort.MultDimToOneDimSort;

public class PKASort extends MultDimToOneDimSort{

	double a,b,c,d,e,f,g;
	
	public PKASort(double a,double b,double c,double d,double e,double f,double g) {
		this.a = a;
		this.b = b;
		this.c = c;
		this.d = d;
		this.e = e;
		this.f = f;
		this.g = g;
	}
	@Override
	public double scoreFunction(double[] v) {
		 double x;
		x =b * v[0];
		x = c * v[1];
		x = d*v[2];
		x = e * v[3];
		x = f*v[4];
		x =  g*v[5];
		return (a + b * v[0] + c * v[1] + d*v[2] + e * v[3] + f*v[4] + g*v[5]);
	}
	
	@Override
	public String toString() {
		return "PKA-Sort linear";
	}

}
