
package flo.optionsArrayRQMC;

import umontreal.ssj.util.sort.MultDimToOneDimSort;

public class HestonModelOneDimSort extends MultDimToOneDimSort {
	
	/**
	 * The dimension of the object to be sorted.
	 */
	protected int dimension = 1;
	double a, b,delta;
	int numSteps;
	

	
	HestonModelOneDimSort(double a, double b, double delta, int numSteps){
		this.a=a;
		this.b=b;
		this.delta=delta;
		this.numSteps = numSteps;
	}
	
	HestonModelOneDimSort(double a, double b, int numSteps){
		this(a,b,0.0,numSteps);
	}
	
	@Override
	public double scoreFunction(double[] v) {
		double score = a*v[0] + v[1] * (b + Math.sqrt(v[2]/(double)numSteps) * delta);
		
		
		
		return score;
	}
	
	@Override
	public String toString() {
		return "Heston-Sort linear";
	}
	
}
