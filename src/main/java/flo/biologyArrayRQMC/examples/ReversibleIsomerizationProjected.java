package flo.biologyArrayRQMC.examples;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import flo.neuralNet.NeuralNet;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.probdist.PoissonDist;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.util.Chrono;

import umontreal.ssj.util.sort.MultiDim01;

public class ReversibleIsomerizationProjected extends ChemicalReactionNetwork implements MultiDim01 {


	double N0; //Total number of molecules
		public ReversibleIsomerizationProjected(double[] c, double[] X0, double tau, double T, double N0) {
			this.c = c;
			this.X0 = X0;
			this.tau = tau;
			this.T = T;
			S = new double[][] { { -1,1}, { 1,-1 }};
			init();
			this.N0 = N0;
		}
		
	

		@Override
		public int compareTo(MarkovChainComparable m, int i) {
			if (!(m instanceof ReversibleIsomerizationProjected)) {
				throw new IllegalArgumentException("Can't compare an ReversibleIso with other types of Markov chains.");
			}
			double mx;

			mx = ((ReversibleIsomerizationProjected) m).X[i];
			return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));
		}

		public String toString() {
			StringBuffer sb = new StringBuffer("----------------------------------------------\n");
			sb.append(" ReversibleIsomerisation:\n");
			sb.append("X0 =\t" + "{" + X0[0] + ", " + (N0-X0[0]) + "}\n");
			sb.append("c =\t" + "{" + c[0] + ", " + c[1] + "}\n");
			sb.append("T =\t" + T + "\n");
			sb.append("tau =\t" + tau + "\n");
			sb.append("steps =\t" + numSteps + "\n");
			sb.append("----------------------------------------------\n\n");

			return sb.toString();
		}

		@Override
		public void computePropensities() {
			double x1 = (N0-X[0]);
			a[0] = c[0] * X[0];
			a[1] = c[1] * x1;
		}

		@Override
		public double getPerformance() {
			return X[0];
		}
		
		
		
		

		

		@Override
		public double[] getPoint() {
			double[] state01 = new double[N];
			for(int i = 0; i < N; i++)
				state01[i] = getCoordinate(i);
			return state01;
		}

		@Override
		public double getCoordinate(int j) {
			double zvalue;
			double x02 = N0 - X0[0];
			switch (j) {
	        case 0:   
	        	zvalue = (X[j]- X0[j]-0.5*c[0]*X0[j]*(X0[j]-1)*X0[j+1]*step+c[1]*step /6.0*X0[j]*(X0[j]-1)*(X0[j]-2)-c[2]*step*x02+c[3]*step*X0[j])/(Math.sqrt(0.5*c[0]*X0[j]*(X0[j]-1)*X0[j+1]*step+c[1]*step /6.0*X0[j]*(X0[j]-1)*(X0[j]-2)+c[2]*step*x02+c[3]*step*X0[j]));
	            return NormalDist.cdf01 (zvalue);
	        case 1:   
	        	zvalue = (X[j]- X0[j]+0.5*c[0]*X0[j-1]*(X0[j-1]-1)*X0[j]*step-c[1]*step /6.0*X0[j-1]*(X0[j-1]-1)*(X0[j-1]-2))/(Math.sqrt(0.5*c[0]*X0[j]*(X0[j]-1)*x02*step+c[1]*step /6.0*X0[j]*(X0[j]-1)*(X0[j]-2)));
	            return NormalDist.cdf01 (zvalue);
	        default:
	            throw new IllegalArgumentException("Invalid state index");
	    	}
//			return NormalDist.cdf01(X[j]);
		}

	

}
