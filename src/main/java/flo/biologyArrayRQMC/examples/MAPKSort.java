package flo.biologyArrayRQMC.examples;

import umontreal.ssj.util.sort.MultDimToOneDimSort;

public class MAPKSort extends MultDimToOneDimSort{

	double [] coeffs;
	boolean bias;
	int[][] varIndices;
	
	
	public MAPKSort(double[] coeffs, int[][] varIndices, boolean bias) {
		this.dimension = 11;
		this.coeffs = new double [coeffs.length];
		for(int i = 0; i < coeffs.length; i++)
			this.coeffs[i] = coeffs[i];
		
		this.bias = bias;
		this.varIndices = new int[varIndices.length][];
		for(int i = 0; i < varIndices.length; i++) {
			this.varIndices[i] = new int[varIndices[i].length];
			for(int j = 0; j < this.varIndices[i].length; j++)
				this.varIndices[i][j] = varIndices[i][j];
		}
	}
	
	@Override
	public double scoreFunction(double[] v) {
		double score = 0.0;
		if(bias) {
			score = coeffs[0];
			for(int j = 0; j < varIndices.length; j++) {
				double temp = 1.0;
				for( int col : varIndices[j])
					temp *= v[col];
				score += coeffs[j+1] * temp;
			}
		}
		else {
			for(int j = 0; j < varIndices.length; j++) {
				double temp = 1.0;
				for( int col : varIndices[j])
					temp *= v[col];
				score += coeffs[j] * temp;
			}
		}
//		score = v[9] + v[10];
		return score;
	}
	
	@Override
	public String toString() {
		return "MAPK-Sort";
	}

}
