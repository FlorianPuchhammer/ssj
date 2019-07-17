package flo.biologyArrayRQMC.examples;

import java.util.Arrays;

import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.markovchainrqmc.MarkovChainComparableWithSubsteps;
import umontreal.ssj.probdist.PoissonDist;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.util.sort.MultiDim;

public abstract class ChemicalReactionNetworkSubsteps extends MarkovChainComparableWithSubsteps implements MultiDim {
	double delta = 0;
	int step; // Current step.

	double[] X0; // Initial data
	double[] a; // Propensity functions
	double[][] S; // Stoichiometric matrix

	double[] c; // reaction rates

	int K; // Number of reactions
	double tau; // Time step
	double T; // Final time
	double[] X; // states, i.e., the number of molecules of each species
	int N; // Number of molecular species
//	public int numSteps;

//	double[][] Stransposed;

	public void init() {
		K = c.length;
		N = X0.length;
		X = new double[N];
		a = new double[K];
		numSteps = (int) Math.ceil(T / tau);
		stateDim = N;
		X = new double[N];
		for (int j = 0; j < N; ++j)
			X[j] = X0[j];
//		Stransposed = transposeMatrix(S);
		numSubsteps = K;
	}

	public void setInitialState(double[] X0) {
		this.X0 = X0;
	}

	public void setNumSteps(int numSteps) {
		this.numSteps = numSteps;
	}

	public int getK() {
		return K;
	}

	private static double[] multMv(double[][] M, double[] v) {
		double[] resu = new double[M.length];
		for (int i = 0; i < M.length; i++) {
			for (int j = 0; j < M[0].length; j++) {
				resu[i] += M[i][j] * v[j];

			}

		}
		return resu;

	}

	// product of two vectors
	private static double multvv(double[] v1, double[] v2) {
		double resu = 0;
		for (int i = 0; i < v1.length; i++)
			resu += v1[i] * v2[i];

		return resu;

	}

	// product of a vector by a constant
	static double[] multvc(double[] v1, double c) {
		double[] resu = new double[v1.length];
		for (int i = 0; i < v1.length; i++)
			resu[i] = v1[i] * c;

		return resu;

	}

	// sum of two vectors
	static double[] sumvv(double[] v1, double[] v2) {
		double[] v = new double[v1.length];
		for (int i = 0; i < v1.length; i++) {
			v[i] = v1[i] + v2[i];
		}
		return v;
	}

	// transpose of matrix
	static double[][] transposeMatrix(double[][] m) {
		double[][] tmp = new double[m[0].length][m.length];
		for (int i = 0; i < m[0].length; i++)
			for (int j = 0; j < m.length; j++)
				tmp[i][j] = m[j][i];
		return tmp;
	}

	// Initial value of X.
	public void initialState() {
		step = 0;
//		X = X0;
		for (int j = 0; j < N; ++j)
			X[j] = X0[j];
	}

	abstract public void computePropensities();


//	 Simulates the next step.
//		public void nextStep(RandomStream stream) {
//			//step+=tau;
//			step++;
//			double[] p = new double[K] ;
//			computePropensities();
//			
////			System.out.println("TEST\tStep: " + step);
//			
//			for ( int k=0; k<K; k++){	
////				System.out.println("TEST\tSpecies: " + k);
//				p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()); 
//			}
//			
//			double [] [] Stransposed = new double[K][N];
//			
//			Stransposed=transposeMatrix(S);
//			
//			double [] [] res= new double[K][N];
//			for ( int k=0; k<K; k++){
//				 res[k]=multvc(Stransposed[k],p[k]);
//				
//				 
//			}
//			double [] r = res[0];
//			for ( int k=1; k<K; k++){
//			r=sumvv( r, res[k]);
//			
//			}
//			
//			X  = sumvv(X,r); 
////			System.out.println("r = " + r[0]);
////			System.out.println("p. = " + (-p[0] + p[1]));
//
//		}	

	@Override
	public int dimension() {
		return N;
	}

	@Override
	public double[] getState() {
		return X;
	}

	@Override
	public void nextStep(RandomStream stream, int substep) {
		System.out.println("called with substep " + substep);
		if (substep == 0) {
			computePropensities();
		}

		double p = PoissonDist.inverseF(a[substep] * tau, stream.nextDouble());

		double[] temp = new double[N];
		for (int j = 0; j < N; j++)
			temp[j] = X[j];

		for (int j = 0; j < N; j++) {
			temp[j] += p * S[j][substep];
		}

		X = temp;

	}

}
