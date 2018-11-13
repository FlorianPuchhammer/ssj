package flo.biologyArrayRQMC.examples;

import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.util.sort.MultiDim01;

public class PKA extends ChemicalReactionNetwork implements MultiDim01 {

	public PKA(double[] c, double[] X0, double tau, double T) {
		this.c = c;
		this.X0 = X0;
		this.tau = tau;
		this.T = T;
		S = new double[][] { { -1, 1, 0, 0, 0, 0 }, { -2, 2, -2, 2, 0, 0 }, { 1, -1, -1, 1, 0, 0 },
				{ 0, 0, 1, -1, -1, 1 }, { 0, 0, 0, 0, 1, -1 }, {0,0,0,0,2,-2} };
		init();
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append(" cAMP activation of PKA:\n");
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
		if (!(m instanceof PKA)) {
			throw new IllegalArgumentException("Can't compare a PKA with other types of Markov chains.");
		}
		double mx;

		mx = ((PKA) m).X[i];
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
		 	
		    	
		    	switch (j) {
		        case 0:   
		        	zvalue = (X[j]- X0[j]+c[0]*0.5*X0[0]*X0[1]*(X0[1]-1)*step-c[1]*X0[2]*step)/(Math.sqrt(c[0]*0.5*X0[0]*X0[1]*(X0[1]-1)*step+c[1]*X0[2]*step));
		            return NormalDist.cdf01 (zvalue);
		        case 1:   
		        	zvalue = (X[j]- X0[j]+c[0]*0.5*X0[0]*X0[1]*(X0[1]-1)*step-c[1]*X0[2]*step+c[2]*0.5*X0[2]*X0[1]*(X0[1]-1)*step-c[3]*X0[3]*step)/(Math.sqrt(c[0]*0.5*X0[0]*X0[1]*(X0[1]-1)*step+c[1]*X0[2]*step+c[2]*0.5*X0[2]*X0[1]*(X0[1]-1)*step+c[3]*X0[3]*step));
		            return NormalDist.cdf01 (zvalue);
		        case 2:  
		        	zvalue = (X[j]- X0[j]-c[0]*0.5*X0[0]*X0[1]*(X0[1]-1)*step+c[1]*X0[2]*step+c[2]*0.5*X0[2]*X0[1]*(X0[1]-1)*step-c[3]*X0[3]*step)/(Math.sqrt(c[0]*0.5*X0[0]*X0[1]*(X0[1]-1)*step+c[1]*X0[2]*step+c[2]*0.5*X0[2]*X0[1]*(X0[1]-1)*step+c[3]*X0[3]*step));
		            return NormalDist.cdf01 (zvalue);
		        case 3:  
		        	zvalue = (X[j]- X0[j]-c[2]*0.5*X0[2]*X0[1]*(X0[1]-1)*step+c[3]*X0[3]*step+c[4]*X0[3]*step-c[2]*0.5*X0[4]*X0[5]*(X0[5]-1)*step)/(Math.sqrt(c[2]*0.5*X0[2]*X0[1]*(X0[1]-1)*step+c[3]*X0[3]*step+c[4]*X0[3]*step+c[2]*0.5*X0[4]*X0[5]*(X0[5]-1)*step));
		            return NormalDist.cdf01 (zvalue);
		        case 4:  
		        	zvalue = (X[j]- X0[j]-c[4]*X0[3]*step+c[5]*0.5*X0[4]*X0[5]*(X0[5]-1)*step)/(Math.sqrt(c[4]*X0[3]*step+c[5]*0.5*X0[4]*X0[5]*(X0[5]-1)*step));
		            return NormalDist.cdf01 (zvalue);
		        case 5:  
		        	zvalue = (X[j]- X0[j]-c[4]*X0[3]*step+c[5]*0.5*X0[4]*X0[5]*(X0[5]-1)*step)/(Math.sqrt(c[4]*X0[3]*step+c[5]*0.5*X0[4]*X0[5]*(X0[5]-1)*step));
		            return NormalDist.cdf01 (zvalue);
		      
		       
		        default:
		            throw new IllegalArgumentException("Invalid state index");
		    	}
	}

	@Override
	public void computePropensities() {
		a[0] = 0.5 * c[0] * X[0] * X[1] * (X[1] - 1.0);
		a[1] = c[1] * X[2];
		a[2] = 0.5 * c[2] * X[2] * X[1] * (X[1] - 1.0);
		a[3] = c[3] * X[3];
		a[4] = c[4] * X[3];
		a[5] = 0.5 * c[5] * X[4] * X[5] * (X[5] - 1.0);
	}

	@Override
	public double getPerformance() {
		return X[0]; //PKA
//		return X[1]; //cAMP
	}

}
