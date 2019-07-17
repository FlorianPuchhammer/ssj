package flo.biologyArrayRQMC.examples;

import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.util.sort.MultiDim01;

public class LinearBirthDeath extends ChemicalReactionNetwork implements MultiDim01 {

	public LinearBirthDeath(double[] c, double[] X0, double tau, double T) {
		this.c = c;
		this.X0 = X0;
		this.tau = tau;
		this.T = T;
		S = new double[][] { { -1.0, 1.0} };
		init();
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append("LinearBirthDeath:\n");
		sb.append("Number of reactions K = " + K + "\n");
		sb.append("Number of species N = " + N + "\n");
		sb.append("X0 =\t" + "{" + X0[0]);
		for(int i = 1; i < X0.length; i++)
			sb.append(", " + X0[i]);
		sb.append("}\n");
		
		sb.append("c =\t" + "{" + c[0]);
		for(int i = 1; i < c.length; i++)
			sb.append(", " + c[i]);
		sb.append("}\n");
		sb.append("T =\t" + T + "\n");
		sb.append("tau =\t" + tau + "\n");
		sb.append("steps =\t" + numSteps + "\n");
		sb.append("----------------------------------------------\n\n");
		return sb.toString();
	}

	@Override
	public int compareTo(MarkovChainComparable m, int i) {
		if (!(m instanceof LinearBirthDeath)) {
			throw new IllegalArgumentException("Can't compare LinearBirthDeath with other types of Markov chains.");
		}
		double mx;

		mx = ((LinearBirthDeath) m).X[i];
		return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));
	}

	@Override
	public double[] getPoint() {
		double[] state01 = new double[N];
    	for(int i=0;i<N;i++)
        state01[i] = getCoordinate(i);       
        return state01;
	}

	@Override
	public double getCoordinate(int j) {
		double zvalue;                 
	 	
//		return X[j];
    	
    	switch (j) {
        case 0:   
        	zvalue = (X[j]- X0[j] - tau * step * (- c[0]*X0[0] + c[1]*X0[0]))/Math.sqrt(tau * step * X[0]*(c[0]*c[0] + c[1]*c[1]));
            
        	return NormalDist.cdf01 (zvalue);
      
      
       
        default:
            throw new IllegalArgumentException("Invalid state index");
    	}
	}

	@Override
	public void computePropensities() {
		a[0] = c[0] * X[0] ;
		a[1] = c[1] * X[0];
//		System.out.println("TestPropensities: " + X0[0] + ", " + X[0]);
	}

	@Override
	public double getPerformance() {
		return X[0]; 

	}

}
