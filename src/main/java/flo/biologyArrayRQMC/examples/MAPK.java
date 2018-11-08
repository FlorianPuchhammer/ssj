package flo.biologyArrayRQMC.examples;

import umontreal.ssj.markovchainrqmc.MarkovChainComparable;

public class MAPK extends ChemicalReactionNetwork {
	// TODO Auto-generated method stub
	public MAPK (double[] c, double[] X0, double tau, double T) {
		this.c = c;
		this.X0 = X0;
		this.tau = tau;
		this.T = T;
		S = new double[][] {
				{-1,1,0,0,0,0,0,0,0,0,0,0,0,1},
				{-1,1,0,1,-1,1,1,0,0,0,0,0,0,0},
				{1,-1,-1,0,0,0,0,0,0,0,0,0,0,0},
				{0,0,1,-1,0,0,0,0,0,0,0,0,0,0},
				{0,0,1,0,-1,1,0,0,0,1,0,-1,1,0},
				{0,0,0,0,1,-1,-1,0,0,0,0,0,0,0},
				{0,0,0,0,0,0,1,-1,1,0,0,0,0,0},
				{0,0,0,0,0,0,0,-1,1,0,1,-1,1,1},
				{0,0,0,0,0,0,0,1,-1,-1,0,0,0,0},
				{0,0,0,0,0,0,0,0,0,1,-1,0,0,0},
				{0,0,0,0,0,0,0,0,0,0,0,1,-1,-1}
		};
		init();
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append("MAPK cascade model:\n");
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
		if (!(m instanceof MAPK)) {
			throw new IllegalArgumentException("Can't compare a MAPK with other types of Markov chains.");
		}
		double mx;

		mx = ((MAPK) m).X[i];
		return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));
	}

	@Override
	public void computePropensities() {
		a[0] = c[0] * X[0] * X[1];
		a[1] = c[1] * X[2];
		a[2] = c[2] * X[2];
		a[3] = c[3] * X[3];
		a[4] = c[4] * X[1]*X[4];
		a[5] = c[5] * X[5];
		a[6] = c[6] * X[6];
		a[7] = c[7] * X[6] * X[7];
		a[8] = c[8] * X[8];
		a[9] = c[9] * X[8];
		a[10] = c[10] * X[9];
		a[11] = c[11] * X[4] * X [7];
		a[12] = c[12] * X[10];
		a[13] = c[13] * X[10];

	}

	@Override
	public double getPerformance() {
		return X[0];
	}

}
