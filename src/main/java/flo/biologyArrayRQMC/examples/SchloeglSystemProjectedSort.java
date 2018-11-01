package flo.biologyArrayRQMC.examples;

import umontreal.ssj.util.sort.MultDimToOneDimSort;

/**
 * Implements the sorting function 
 * \f[
 * a + b * \textrm{tanh}(c x + d y + e)
 * \f]
 * @author florian
 *
 */
public class SchloeglSystemProjectedSort extends MultDimToOneDimSort {

	double a,b,c,d,e;
	public SchloeglSystemProjectedSort(double a, double b, double c, double d, double e) {
		this.a = a;
		this.b = b;
		this.c = c;
		this.d = d;
		this.e = e;
	}
	@Override
	public double scoreFunction(double[] v) {		
		return (  a + b * Math.tanh( c*v[0] + d*v[1] + e)   );
	}

}
