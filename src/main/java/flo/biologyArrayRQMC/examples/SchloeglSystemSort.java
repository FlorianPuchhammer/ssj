package flo.biologyArrayRQMC.examples;

import umontreal.ssj.util.sort.MultDimToOneDimSort;

public class SchloeglSystemSort extends MultDimToOneDimSort {

	public SchloeglSystemSort() {
		this.dimension = 3;
	}
	
	public void projection(double[] x, double[] y) {
		y[0] = x[0];
		y[1] = 300250.0 - x[0] - x[1];
	}

	@Override
	public double scoreFunction(double[] u) {
		double[] v = new double[u.length - 1];
		projection(u, v);
		double x = v[0];
		double y = v[1];
		return 205453.08151866705 - 11.035377136126005 * x + 0.007381380440287618 * x * x
				- 3.3887179199387418E-6 * x * x * x - 3.0049650331867643 * y + 0.00011740773224161276 * x * y
				- 2.1797523502454592E-8 * x * x * y + 0.00001464519923613829 * y * y
				- 3.0363962286145344E-10 * x * y * y - 2.3778485538469843E-11 * y * y * y;

	}
	
	public String toString() {
		return "SchloeglSystemSort";
	}

}
