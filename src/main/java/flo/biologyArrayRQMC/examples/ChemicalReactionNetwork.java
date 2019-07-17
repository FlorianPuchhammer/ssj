package flo.biologyArrayRQMC.examples;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import flo.neuralNet.NeuralNet;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.probdist.PoissonDist;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.util.Chrono;
import umontreal.ssj.util.sort.MultiDim;

public abstract class ChemicalReactionNetwork extends MarkovChainComparable implements MultiDim {
	double delta = 0;
	int step; // Current step.

	double[] X0; // Initial data
	double[] a; // Propensity functions
	double[][] S; // Stoichiometric matrix

	double[] c; // reaction rates

	public int K; // Number of reactions
	double tau; // Time step
	double T; // Final time
	double[] X; // states, i.e., the number of molecules of each species
	int N; // Number of molecular species
//	public int numSteps;

	public void init() {
		K = c.length;
		N = X0.length;
		X = new double[N];
		a = new double[K];
		numSteps = (int) Math.ceil(T / tau);
		stateDim = N;
		X = new double[N];
		for(int j = 0; j < N; ++j)
			X[j] = X0[j];
//		initialState();
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
		delta = 0;
//		X = X0;
		for(int j = 0; j < N; ++j)
			X[j] = X0[j];
	}

	abstract public void computePropensities();

	public void nextStep(RandomStream stream, double[] state, int reps) {
		// write state before step
		double[] tmp = getState();
		for (int d = 0; d < tmp.length; d++)
			state[d] = tmp[d];

		double[] futureState = new double[state.length];
		nextStep(stream);
		for (int j = 0; j < state.length; j++)
			futureState[j] = (getState())[j];

		for (int r = 1; r < reps; r++) {
			// TODO: for sobol, etc. need to reset stream to prev. point.
			
			// reset X to old state
			for (int j = 0; j < state.length; j++)
				X[j] = state[j];
			

			// carry out next step
			step--;
			nextStep(stream);

			// update the future state avg.
			for (int j = 0; j < state.length; j++) {
				futureState[j] = ((double) r) * futureState[j] + (getState())[j];
				futureState[j] /= (double) (r + 1.0);
//				futureState[j] += (getState())[j];
			}

		}
//		// after all the repetitions, set the state to the avg
//		for (int j = 0; j < state.length; j++)
//			X[j] = futureState[j] ;// / (double) reps;
	}

	public void simulSteps(int numSteps, RandomStream stream, double[][] states, int reps) {
		initialState();
		this.numSteps = numSteps;
		int step = 0;
		while (step < numSteps && !hasStopped()) {
			states[step] = new double[getState().length];
			nextStep(stream, states[step], reps);
			++step;
		}
	}

	public void simulRuns(int n, int numSteps, RandomStream stream, double[][][] states, double[] performance,
			int reps) {
		for (int i = 0; i < n; i++) {
			states[i] = new double[numSteps][];
			simulSteps(numSteps, stream, states[i], reps);
			performance[i] = getPerformance();
		}
	}

//	 Simulates the next step.
//		public void nextStep(RandomStream stream) {
//			//step+=tau;
//			System.out.println("X_j = " + X[0]);
//
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
////			System.out.println("999 ??? " + (X[0]+p[0] - p[1]));
//			
//			X  = sumvv(X,r); 
//			System.out.println("X = " + X[0]);
//			delta += (-p[0]+p[1]);
//			System.out.println("Delta = " + delta);
//			System.out.println("p. = " + (-p[0] + p[1]));
//
//		}	
	
	@Override
	public void nextStep(RandomStream stream) {
		step++;
//		System.out.println("X_j = " + X[0]);
		double[] p = new double[K];
		computePropensities();

//			System.out.println("TEST\tStep: " + step);

		for (int k = 0; k < K; k++) {
//				System.out.println("TEST\tSpecies: " + k);
			p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()); 
//			System.out.println("TEST" + k + ": a =\t" + p[k]);

		}
//		System.out.println("TEST X "  + ": X =\t" + X[0]);

//		double temp = 0;
		double [] temp = new double[N];
		for(int j = 0; j < N; j++)
			temp[j] = X[j];
		for (int n = 0; n < N; n++) {
			
			for (int k = 0; k < K; k++) {
				temp[n] +=   S[n][k] * p[k];
//				System.out.println("TEST" + k + ": p =\t" + S[n][k]);
			}
			
//			X[n] += temp;
			
		}
		X = temp;
//		delta += (-p[0]+p[1]);
//		System.out.println("X_{j+1} = " + X[0]);
//		System.out.println("p:\t" + (S[0][0] * p[0]+ S[0][1] * p[1]));
//		System.out.println("Delta = " + delta);

		
	}

	@Override
	public int dimension() {
		return N;
	}

	@Override
	public double[] getState() {
		return X;
	}

	public void genData(String dataPath, String dataLabel, int n, int numSteps, RandomStream stream, int reps)
			throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		simulRuns(n, numSteps, stream, states, performance, reps);
		StringBuffer sb;
		FileWriter fw;
		File file;
		for (int step = 0; step < numSteps; step++) {
			sb = new StringBuffer("");
			file = new File(dataPath + dataLabel + "_Step_" + step + ".csv");
//			file.getParentFile().mkdirs();
			fw = new FileWriter(file);

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < getStateDimension(); j++)
					sb.append(states[i][step][j] + ",");
				sb.append(performance[i] + "\n");
			}
			fw.write(sb.toString());
			fw.close();
			System.out.println("*******************************************");
			System.out.println(" STEP " + step);
			System.out.println("*******************************************");
			System.out.println(sb.toString());
		}

	}

	public static void main(String[] args) throws IOException {
		ChemicalReactionNetwork model;
		/*
		 * ******************* PKA
		 **********************/
//		double[]c = {8.696E-5, 0.02, 1.154E-4,0.02,0.016,0.0017};//Nano: 1E-9
//		double[] x0 = {33000.0,33030.0, 1100.0, 1100.0, 1100.0, 1100.0};
//		double T = 0.00005;
//		double tau = T/15.0;
//
//		
//		
//		 model = new PKA(c,x0,tau,T);
////		String dataFolder = "data/PKA/";
//		String dataFolder = "";
//		model.init();
		
		/* *******************
		 * REVERSIBLE ISO
		 **********************/
//		double alpha = 1E-4;
//		double[]c = {1.0,alpha};
//		double[] x0 = {1.0E2,1.0E6};
//		double T = 1.6;
//		double tau = T/8.0;
//
//		
//		
//		 model = new ReversibleIsomerization(c,x0,tau,T);
//		 model.init();

		/*
		 * ******************* MAPK
		 **********************/
		double[] c = { 0.027, 1.35, 1.5, Math.log(2.0) * 10.0, 0.028, 1.73, 15.0, 0.027, 1.35, 1.5,
				Math.log(2.0) * 10.0, 0.028, 1.73, 15.0 };// Nano: 1E-9
		double[] x0 = { 2500.0, 4500.0, 4500.0, 4500.0, 7000.0, 8000.0, 2500.0, 4500.0, 4500.0, 5500.0, 7000.0 };
		double T = 0.01;
		double tau = T / 10.0;
		model = new MAPK(c, x0, tau, T);
		String dataFolder = "";
		model.init();

		/*
		 * ******************* ENZYME-KINETICS
		 **********************/
//		double[]c = {1E-3,1E-4, 0.1};//Nano: 1E-9
//		double[] x0 = {200,500,200,0};
//		double T = 10.0;
//		double tau = T/7.0;
//
//		 model = new EnzymeKinetics(c,x0,tau,T);
//		String dataFolder = "data/EnzymeKineticsP/";
//		String dataFolder = "";
//
//		 model.init();

//		 /* *******************
//		  * SCHLOEGL SYSTEM
//		  **********************/
//		double[] c = { 3E-7, 1E-4, 1E-3, 3.5 };
//		double[] x0 = { 250.0, 1E5};
//		double N0 = 2E5 + 1E5 + 250.0;
//		double T = 4.0;
//		double tau = T/15.0;
//
//		model = new SchloeglSystemProjected(c, x0, tau, T,N0);
////			String dataFolder = "data/Schloegl/";
//			String dataFolder = "";
//
//		model.init();

		/*
		 * ******************* REVERSIBLE ISOMERIZATION Projected
		 **********************/
//		double epsInv = 1E2;
//		double alpha = 1E-4;
//		double[]c = {1.0,alpha};
//		double[] x0 = {epsInv};
//		double N0 = epsInv + epsInv/alpha;
//		double T = 1.6;
//		double tau = 0.2;
//
//		
//		
//		 model = new ReversibleIsomerizationProjected(c,x0,tau,T,N0);
//			String dataFolder = "data/RevIso/";
//			String dataFolder = "";

//			 model.init();

		System.out.println(model.toString());

		int numChains = 524288 * 2  /4; 
		int logNumChains = 19 + 1;
		int reps = 512;  // reps = 1; 

		Chrono timer = new Chrono();
		RandomStream stream = new MRG32k3a();

//		String dataLabel = "MCData";		
//		NeuralNet.genData(model,dataFolder,dataLabel, numChains, model.numSteps, stream);

		String dataLabel = "MCDataLessNoise"; //  dataLabel = "MCData";

		model.genData(dataFolder, dataLabel, numChains, model.numSteps, stream, reps);

		System.out.println("\n\nTiming:\t" + timer.format());
	}

}
