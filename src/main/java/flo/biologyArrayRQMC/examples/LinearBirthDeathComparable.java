package flo.biologyArrayRQMC.examples;


// For Asian option pricing, state must contain:
// (1) state of Brownian motion;  (2) current sum for the final average.

import umontreal.ssj.rng.*;
import umontreal.ssj.markovchainrqmc.*;
import umontreal.ssj.util.sort.MultiDim;
import umontreal.ssj.probdist.PoissonDist;

class LinearBirthDeathComparable extends MarkovChainComparable   implements MultiDim{


	int step; // Current step.

	double [] X0; // Initial data 
	double[] a; // Propensity functions 
	double[][] S ; // Stoichiometric matrix
	int c;
	int K; // Number of reactions
	double tau ; // Time step
	int T; // Final time
	double [] X; //states, i.e., the number of molecules of each species
	int N; //Number of molecular species 
	//double[][] S={{-1.0,1.0}};
	
	public LinearBirthDeathComparable(int K, int c, int N, double []  X0 , double[][] D, double tau, int T) {
		this.K = K;
		this.X0 = new double[N];
		this.X0 = X0;
		this.a = new double[K];
			
		this.S = new double [N][K];
		this.S = D;
		this.N = N;		
		this.X = new double[N] ;		
		this.T= T;
		this.c = c;
		this.tau = tau;		
		stateDim = N;
		
	}
	

	// product of a vector by a constant 
	double[]  mutvc( double [] v1, double c){
		double[] resu = new double[v1.length];
		 for (int i=0; i<v1.length; i++)		  
		      resu[i]= v1[i]*c;		     		   		  
		  
		 return resu;
		 
	}
	// sum of two vectors
	double [] sumvv( double [] v1, double[] v2){
		double[] v = new double [v1.length];
	for(int i = 0; i < v1.length; i++) {
		  v[i] = v1[i] + v2[i];
		}
	return v;
	}
		// transpose of matrix
	public static double[][] transposeMatrix(double [][] m){
		double [][] tmp=new double[m[0].length][m.length];
 	for (int i=0; i< m[0].length; i++) 
 		for (int j=0; j< m.length; j++) 
 		tmp[i][j] = m[j][i];
 	return tmp;
	}
//	public double calcFractionPositivePayoff(int n) {
//		TallyStore statRuns = new TallyStore();
//		simulRunsWithSubstreams(n, T, new MRG32k3a(), statRuns);
//		statRuns.quickSort();
//		return (n - statRuns.getDoubleArrayList().lastIndexOf(0.0) - 1.0) / n;
//	}

	// Initial value of X.
	public void initialState() {
		step = 0;
		X = X0;		
	}

	// Simulates the next step.
	public void nextStep(RandomStream stream) {
		
		step++;
		double[] p = new double[K] ;
		for ( int k=0; k<K; k++){
			a[k] =c*X[0];
		  p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()) ;
		 
		}
		

		
		
		
		double [] [] G = new double[K][N];
		
		G=transposeMatrix(S);
		

		double [] [] ress= new double[K][N];
		for ( int k=0; k<K; k++){
			 ress[k]=mutvc(G[k],p[k]);
			
			 
		}
		double [] r = ress[0];
		for ( int k=1; k<K; k++){
		r=sumvv( r, ress[k]);
		
		}
		
		X  = sumvv(X,r); 
	
									
	}

	// Returns the statistics (performance measure) accumulated so far.
	// Valid only after last step (at time t).
	public double getPerformance() {
		
		//return X[0]*X[0];
		return X[0];
		
	}

	

	public int compareTo(MarkovChainComparable m, int i) {
		if (!(m instanceof LinearBirthDeathComparable)) {
			throw new IllegalArgumentException("Can't compare a " + "Biology example with other types of Markov chains.");
		}
		double  mx;
		
			mx = ((LinearBirthDeathComparable) m).X[i];
			return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));
		
		}	
	

   
	public int dimension() {
    	return N;
     }


	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append("LinearBirthDeath:\n");
		for (int n=0; n<N; n++){
		//sb.append(" Biology example:\n");
		//sb.append(" X() : number of molecules of types k                  = " + PrintfFormat.format(8, 3, 1, X[n]) + "\n");
		
		}
		return sb.toString();
	}

	@Override
	public double[] getState() {
		return X;
	}
}
