package flo.biologyArrayRQMC.examples;

import umontreal.ssj.util.sort.MultDimToOneDimSort;

public class PKASort extends MultDimToOneDimSort{

	double [] coeffs;
	boolean bias;
	
	
	
	public PKASort(double[] coeffs) {
		this.dimension = 6;
		this.coeffs = new double [coeffs.length];
		for(int i = 0; i < coeffs.length; i++)
			this.coeffs[i] = coeffs[i];
		
		bias = !(coeffs.length == dimension);
	}
	
	@Override
	public double scoreFunction(double[] v) {
		double score = 0.0;
		if(bias) {
			score = coeffs[0];
			for(int j = 0; j < v.length; j++)
				score += coeffs[j+1] * v[j];
		}
		else {
			for(int j = 0; j < v.length; j++)
				score += coeffs[j] * v[j];
		}
		
		return score;
	}
	
	@Override
	public String toString() {
		return "PKA-Sort linear";
	}

}
