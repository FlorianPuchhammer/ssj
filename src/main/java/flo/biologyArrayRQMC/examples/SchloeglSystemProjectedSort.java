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
//		return (  a + b * Math.tanh( c*v[0] + d*v[1] + e)   );
		double x = v[0];
		double y = v[1];
		return -52292.5234759465 + 20.4225689154585* x - 
				 0.0017316718020435034* x * x - 3.308787724980430E-6 * (x*x*x) + 
				 1.5531050703226243* y - 0.0004038970641477818* x *y + 
				 4.774206369852412E-8 *(x *x* y) - 0.000015353453902356927 * y *y + 
				 2.0256872317222477E-9*( x *y*y) + 5.055655087428423E-11 * (y*y*y);

	}

}
