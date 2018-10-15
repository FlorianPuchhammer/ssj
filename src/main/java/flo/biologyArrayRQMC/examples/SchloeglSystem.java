package flo.biologyArrayRQMC.examples;

import umontreal.ssj.markovchainrqmc.MarkovChainComparable;

public class SchloeglSystem extends ChemicalReactionNetwork {

	public SchloeglSystem(double[] c, double[] X0, double tau, double T) {
		this.c = c;
		this.X0 = X0;
		this.tau = tau;
		this.T = T;
		S = new double[][] { { 1, -1, 1, -1 }, { -1, 1, 0, 0 }, { 0, 0, -1, 1 } };
		init();
	}

	@Override
	public int compareTo(MarkovChainComparable m, int i) {
		if (!(m instanceof SchloeglSystem)) {
			throw new IllegalArgumentException(
					"Can't compare an SchloeglSystem with other types of Markov chains.");
		}
		double mx;

		mx = ((ReversibleIsomerizationComparable) m).X[i];
		return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append(" ReversibleIsomerisation:\n");
		sb.append("X0 =\t" + "{" + X0[0] + ", " + X0[1] + ", " + X0[2] + "}\n");
		sb.append("c =\t" + "{" + c[0] + ", " + c[1] + ", " + c[2] + "}\n");
		sb.append("T =\t" + T + "\n");
		sb.append("tau =\t" + tau + "\n");
		sb.append("steps =\t" + numSteps + "\n");
		sb.append("----------------------------------------------\n\n");

		return sb.toString();
	}

	@Override
	public void computePropensities() {
		a[0] = 0.5 * c[0] * X[0] * (X[0] - 1.0) * X[1];
		a[1] = c[1] * X[0] * (X[0] - 1.0) * (X[0] - 2.0) / 6.0;
		a[2] = c[2] * X[2];
		a[3] = c[3] * X[0];
	}

	@Override
	public double getPerformance() {
		return X[0];
	}
	
	

}
