package flo.biologyArrayRQMC.examples;

import umontreal.ssj.util.sort.MultDimToOneDimSort;

public class PKASortConditionalLinear extends MultDimToOneDimSort{

	double [] a;
	double [] c;
	int K,N;
	double deltaT;
	
	double[][] S = { { -1, 1, 0, 0, 0, 0 }, { -2, 2, -2, 2, 0, 0 }, { 1, -1, -1, 1, 0, 0 },
		{ 0, 0, 1, -1, -1, 1 }, { 0, 0, 0, 0, 1, -1 }, {0,0,0,0,2,-2} };
	
	public PKASortConditionalLinear(double [] c, int N, double deltaT) {
		this.c = c;
		K = c.length;
		a = new double[K];
		this.dimension= N;
		this.deltaT = deltaT;
		
	}
	@Override
	public double scoreFunction(double[] v) {
		 computePropensities(v);
		 
		 double [] [] Stransposed = new double[K][N];
			
			Stransposed=ChemicalReactionNetwork.transposeMatrix(S);
			
			double [] [] res= new double[K][N];
			for ( int k=0; k<K; k++){
				 res[k]=ChemicalReactionNetwork.multvc(Stransposed[k],a[k] * deltaT);
				
				 
			}
			double [] r = res[0];
			for ( int k=1; k<K; k++){
			r=ChemicalReactionNetwork.sumvv( r, res[k]);
			
			}
			
		 
		return (v[0] + r[0]);
	}
	
	private void computePropensities(double [] v) {
		a[0] = 0.5 * c[0] * v[0] * v[1] * (v[1] - 1.0);
		a[1] = c[1] * v[2];
		a[2] = 0.5 * c[2] * v[2] * v[1] * (v[1] - 1.0);
		a[3] = c[3] * v[3];
		a[4] = c[4] * v[3];
		a[5] = 0.5 * c[5] * v[4] * v[5] * (v[5] - 1.0);
	}
	
	
	
	@Override
	public String toString() {
		return "PKA-Sort-Conditional-linear";
	}

}
