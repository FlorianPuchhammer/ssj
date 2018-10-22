package flo.lacZ;



import umontreal.ssj.rng.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import cern.colt.bitvector.BitVector;
import umontreal.ssj.markovchainrqmc.*;
import umontreal.ssj.util.PrintfFormat;
import umontreal.ssj.util.sort.MultiDim;
import umontreal.ssj.util.sort.MultiDim01;
import umontreal.ssj.probdist.DiscreteDistribution;
import umontreal.ssj.probdist.DiscreteDistributionInt;
import umontreal.ssj.probdist.ExponentialDist;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.probdist.PoissonDist;
import umontreal.ssj.probdist.UniformDist;
import umontreal.ssj.stat.TallyStore;

class cAMPComparable extends MarkovChainComparable  implements MultiDim01 , MultiDim{

	
	
	static interface RealsTo01Map {
        public double realTo01(double x);
    }
    static class NormalCDFMap implements RealsTo01Map {
         public double realTo01(double x) {
            return NormalDist.cdf01(x);
        }
    }
    /** Logistic function with the parameterization from Gerber & Chopin (2015).
     *  f(x) = 1 / [1 + exp(-(x - x1) / (x2 - x1))]
     */
    static class LogisticGCMap implements RealsTo01Map {
        private double x1;
        private double x2;
        public LogisticGCMap(double x1, double x2) {
            this.x1 = x1;
            this.x2 = x2;
        }
         public double realTo01(double x) {
            return 1.0 / (1.0 + Math.exp((x1 - x) / (x2 - x1)));
        }
    }
    /** Logistic function with a centered parameterization.
     *  f(x) = 1 / [1 + exp(-(x - x0) / w)]
     */
    static class LogisticMap implements RealsTo01Map {
        private double x0;
        private double w;
        public LogisticMap(double x0, double w) {
            this.w = w;
            this.x0 = x0;
        }
         public double realTo01(double x) {
            return 1.0 / (1.0 + Math.exp(-(x - x0) / w));
        }
   }

    int step; // Current step.

	double [] X0; // Initial data 
	double[] a; // Propensity functions 
	double[][] S ; // Stoichiometric matrix
	double [] c;
	int K; // Number of reactions
	double tau ; // Time step
	int T; // Final time
	double [] X, Xr, Xold,Xprop;
	int N; //Number of molecular species 
	//double[][] S={{-1.0,1.0}};
	
	double epsilon;
	RealsTo01Map map;       // One-dimensional map from reals to [0,1]
	int [] hor, nuHor;
	private BitVector criticals			= null;
	private int 	nCritical 			= 10;
	Map<Integer,Integer>[] reactantHistos;
	Map<Integer,Integer>[] productHistos;
	private int[][] v	= null;
	int[][] changes	;
	int[][] nuChanges	;
	private int[] g;
	private double[] mu;
	private double[] sigma;
	int [][] Reactant = new int[K][];
	int [][] nuReactant = new int[K][];
	int [][] Product = new int [K][];
	int [][] nuProduct = new int [K][];
	
	private double 	useSimpleFactor 	= 10;
	boolean isNegative = false;
	static double a_sum, a_sum_c;
	private double 	langevinThreshold 	= Double.POSITIVE_INFINITY;
	private int 	numSimpleCalls 		= 100;
    static 	int [] [] G ;
    
    private int[][] reactantHistosKeys = null;
	private int[][] reactantHistosVals = null;
	public cAMPComparable(int K, int N, double[] c, double []  X0 , double tau, int[][] Reactant, int [][] Product,int[][] nuReactant , int[][] nuProduct,int T, RealsTo01Map map, double epsilon) {
		this.K = K;
		this.X0 = new double[N];
		this.X0 = X0;
		this.a = new double[K];
		this.c = c;		
		this.S = new double [N][K];
	
		this.N = N;		
		this.X = new double[N] ;	
		this.Xold = new double[N] ;	
		this.Xprop = new double[N] ;
		for (int i=0;i< N; i++)
			Xprop[i] = 0;
		this.Xr = new double[N] ;
		this.T= T;
		this.tau = tau;		
		stateDim = N;
		this.map =  map;	
		this.epsilon = epsilon;
		this.hor = new int [N]; 
		this.nuHor = new int [N];
		mu = new double[N];
		sigma = new double[N];
		this.Reactant = Reactant;
		this.nuReactant = Reactant;
		this.Product = Product;
		this.nuProduct = Product;
		reactantHistos = new Map[K];
		/*changes = new int[K][Reactant.length+Product.length];
		nuChanges = new int[K][Reactant.length+Product.length];*/
		changes = new int[K][];
		nuChanges = new int[K][];
		/*changes= new ArrayList<><>();
		nuChanges= new ArrayList<><>();*/
		 G = new int[K][N];
		
	
		for (int i=0; i<reactantHistos.length; i++)
			reactantHistos[i] = createHistogramAsMap(Reactant[i]);
		productHistos = new Map[K];
		for (int i=0; i<productHistos.length; i++)
			productHistos[i] = createHistogramAsMap(Product[i]);
		
		v = new int[N][K];
		for (int species=0; species<v.length; species++) {
			for (int reaction=0; reaction<v[species].length; reaction++) {
				if (reactantHistos[reaction].containsKey(species))
					v[species][reaction]-=reactantHistos[reaction].get(species);
				if (productHistos[reaction].containsKey(species))
					v[species][reaction]=productHistos[reaction].get(species);
				if (reactantHistos[reaction].containsKey(species) && productHistos[reaction].containsKey(species) )
					v[species][reaction]=productHistos[reaction].get(species) - productHistos[reaction].get(species);
				
			}
		}
		for (int species=0; species<v.length; species++) {
			for (int k=0; k<K; k++) {
				System.out.print(v[species][k]);
			}
			System.out.println("");
		}
		calculateG();
		
		reactantHistosKeys = new int[Reactant.length][];
		reactantHistosVals = new int[Reactant.length][];
		
		for (int i=0; i<Reactant.length; i++) {
			Map<Integer,Integer> reactantHisto = createHistogramAsMap(Reactant[i]);
			reactantHistosKeys[i] = new int[reactantHisto.size()];
			reactantHistosVals[i] = new int[reactantHisto.size()];
			int index = 0;
			for (int r : reactantHisto.keySet()) {
				reactantHistosKeys[i][index] = r;
				reactantHistosVals[i][index] = reactantHisto.get(r);
				index++;
			}
		}
		
			
	}
	
	public static int faculty(int i) {
		return i==1 ? 1 : i*faculty(i-1);
	}

	public double calculatePropensity(int reaction, double[] X) {
		double re = c[reaction];
		/*double volume = sim.getVolume();
		if (volume>0)
			re = getConstantFromDeterministicRateConstant(re, reaction, volume);*/
		for (int i=0; i<reactantHistosKeys[reaction].length; i++) { //int r : reactantHistos[reaction].keySet()) {
			int freq = reactantHistosVals[reaction][i];
			int r = reactantHistosKeys[reaction][i];
			for (int f=0; f<freq; f++) {
				re*=((double)X[r]-f);
			}
			re/=faculty(freq);
		}
		if (re<0) throw new RuntimeException("Propensity < 0");
		return Math.abs(re);
	}
	public double[] calculatePropensity( double[] X) {
	a_sum = 0;		
	
	for( int reaction =0; reaction<K; reaction++)
	a[reaction] =calculatePropensity(reaction, X) ;
    for (int i=0; i<a.length; i++) {
		
		a_sum += a[i];
	}
    return a;
	}
	public static Map<Integer,Integer> createHistogramAsMap(int[] a) {
		Map<Integer,Integer> re = new HashMap<Integer, Integer>();
		if (a==null || a.length==0) return re;
		
		for (int i=0; i<a.length; i++) {
			if (!re.containsKey(a[i]))
				re.put(a[i], 0);
			re.put(a[i], re.get(a[i])+1);
		}
		return re;
	}
	
 
 void computePropensities()
	{
	 a_sum = 0;	
		int nu;
		double x;
		double num, denom;

		double p;

		

		
		for (int ir = 0; ir < K; ++ir)
		{

			int[]  reactants		= Reactant[ir];
			int[]   nu_reactants	= nuReactant[ir];
			p = c[ir];


			for (int s = 0; s < reactants.length; ++s)
			{
				nu		= nu_reactants[s];
				x		= X[ reactants[s] ];
				num		= x;
				denom	= nu;

				while ((--nu)>0)
				{
					denom	*= nu;
					num		*= (x - nu);
				}

				p *= ((double)num/(double)denom) ;
			}


			a[ir] = p;



		}
     for (int i=0; i<a.length; i++) 
			a_sum += a[i];

	}
 
 private int computeL(int reaction) {
		int firings = Integer.MAX_VALUE;
		
		Map<Integer,Integer> reactantHisto = reactantHistos[reaction];
		for (Integer reactant : reactantHisto.keySet()) 
			//if ( getV(reactant,reaction) !=0)
		//	firings = Math.min(firings,(int)Math.floor(X[reactant])/Math.abs(getV(reactant,reaction)));
				firings = Math.min(firings,(int)Math.floor(X[reactant])/Math.abs(reactantHisto.get(reactant)));
		
		return firings;
	}
 
 private void identifyCriticals() {
		if (criticals==null)
			criticals = new BitVector(K);
		else
			criticals.clear();
		
		for (int j=0; j<criticals.size(); j++)
			if (a[j]>0 && computeL(j)<nCritical )
				criticals.set(j);
	}
 private void preprocessNonCriticals(BitVector criticals) {
		for (int i=0; i<N; i++){
			mu[i]=0;
			sigma[i]=0;
			if (g[i]==0) continue;
			for (int j=0; j<K; j++) {
				if (criticals.get(j) || getV(i,j)==0) continue;
				mu[i]+=getV(i, j)*a[j];
				sigma[i]+=getV(i, j)*getV(i, j)*a[j];
			}
		}
	}
 private void calculateG() {
		
		
		
		g = new int[N];
		
		int[] HOR = new int[N];
		for (int r=0; r<K; r++) {
			int[] reactants = Reactant[r];
			for (int i : reactants){
				HOR[i] = Math.max(HOR[i], reactants.length);
				
			}
		}
		for  ( int i=0; i<HOR.length; i++)
		System.out.println(HOR[i]);
		
		for (int i=0; i<g.length; i++)
			g[i] = HOR[i];
		
		for (int r=0; r<K; r++) {
			for (int reactantSpecies : reactantHistos[r].keySet())
				if (HOR[reactantSpecies]==2 && reactantHistos[r].size()==1 && reactantHistos[r].get(reactantSpecies)==2)
					g[reactantSpecies]=-2;
				else if (HOR[reactantSpecies]==3 && reactantHistos[r].size()==2 && reactantHistos[r].get(reactantSpecies)==2)
					g[reactantSpecies]=-4;
				else if (HOR[reactantSpecies]==3 && reactantHistos[r].size()==1 && reactantHistos[r].get(reactantSpecies)==3)
					g[reactantSpecies]=-3;
		}
	}
 protected double chooseTauNonCriticals(BitVector criticals) {
		preprocessNonCriticals(criticals);
		
		double tau = Double.POSITIVE_INFINITY;
		
		for (int i=0; i<N; i++) {
			
			if (g[i]==0) continue;
			double gi = g[i];
			if (gi==-2)
				gi=2.0+1.0/(X[i]-1.0);
			else if (gi==-3)
				gi=3.0+1.0/(X[i]-1.0)+2.0/(X[i]-2.0);
			else if (gi==-4)
				gi=1.5*(2.0+1.0/(X[i]-1.0));
			
			double max = Math.max(epsilon*X[i]/gi, 1);
			tau = Math.min(tau, max/Math.abs(mu[i]));
			tau = Math.min(tau, max*max/sigma[i]);
			
		}
		return tau;
	}
 public double chooseTauCriticals(BitVector criticals, RandomStream stream) {
		//double a_sum_c = 0;
		for (int i=0; i<K; i++)
			if (criticals.get(i))
				a_sum_c += a[i];
		
		return ExponentialDist.inverseF(1/a_sum_c,stream.nextDouble());
		
	}
 public void performSSA(int numStep, RandomStream stream){
	 int index;
	 double t=0;
	 X =X0;
	 while (t< numStep){
		// CalculatePropensities();
		 a =calculatePropensity( X);
		// computePropensities();
		 //System.out.println("a_sum"+a_sum);
		 tau =  ExponentialDist.inverseF(1/a_sum,stream.nextDouble());
		 double[]  pr = new double[K];
			for (int j=0; j<K;j++){
				
				 pr[j]= a[j]/a_sum;
				// System.out.println("probability"+pr[j]);
			 }
			
			 index = indexprobability(pr);
			 Xr = sumvv(X,G[index]);
			 X = Xr;
			//t++;
			 t = t+tau;
		 
	 }
 }
 
 /*public ArrayList<Integer>  listOfCriticalReactions( ){
		
     double Lj;
     ArrayList<Integer>  critical_reactions = new ArrayList<>() ;
		int Nc = 100;

		for (int k = 0; k < K; ++k)			
		{		
			if (S[0][k]>0) break;
				Lj =  X[0]/ Math.abs(S[0][k]);			
			Lj = Integer.MAX_VALUE;
			if (a[k] > 0.0)
			{
				
				for (int is = 0; is < N ; ++is)
				{							
					if (S[is][k]>0) break;
					Lj = min(Lj, X[is]/ Math.abs(S[is][k]) );
					
				}
				if (Lj < Nc){
					critical_reactions.add(k);			   
				}
			}
			
		}

		return critical_reactions;
	}

 
 void computeHor()
 {
 	for (int numbS = 0; numbS < N; ++numbS)
 	{
 		hor[numbS] = 0;
 		nuHor[numbS] = 0;
 	}

 	int numberOfReactions = K;
 	for (int ir = 0; ir < numberOfReactions; ++ir)
 	{
 		

 		int [] reactantsVector  = Reactant[ir];
 		int [] nuReactantsVector  = nuReactant[ir];
 		int order = 0;
 		for (int is = 0; is < reactantsVector.length; ++is)
 		{
 			order += nuReactantsVector[is];
 		}
 		for (int is = 0; is < reactantsVector.length; ++is)
 		{
 			if (order > hor[reactantsVector[is]])
 			{
 				hor[reactantsVector[is]] = order;
 				nuHor[reactantsVector[is]] = nuReactantsVector[is];
 			}
 		}

 	}

 }*/
 
	
	double [] mutMv( double [] [] M, double[] v){
		 double [] resu = new double [M.length];
		 for (int i=0; i<M.length; i++)
		  {
		    for (int j=0; j<M[0].length; j++)
		    {
		      resu[i] += M[i][j] * v[j];
		     
		    }
		   
		  }
		 return resu;
		 
	}
	// product of two vectors
	double  mutvv( double [] v1, double[] v2){
		double resu=0;
		 for (int i=0; i<v1.length; i++)		  
		      resu+= v1[i]* v2[i];		     		   		  
		  
		 return resu;
		 
	}
	// product of a vector by a constant 
	double[]  mutvc( double [] v1, double c){
		double[] resu = new double[v1.length];
		 for (int i=0; i<v1.length; i++)		  
		      resu[i]= v1[i]*c;		     		   		  
		  
		 return resu;
		 
	}
	double[]  mutvc( int [] v1, double c){
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
	double [] sumvv( double [] v1, int[] v2){
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
	public static int[][] transposeMatrix(int [][] m){
		int [][] tmp=new int[m[0].length][m.length];
 	for (int i=0; i< m[0].length; i++) 
 		for (int j=0; j< m.length; j++) 
 		tmp[i][j] = m[j][i];
 	return tmp;
	}
	public double calcFractionPositivePayoff(int n) {
		TallyStore statRuns = new TallyStore();
		simulRunsWithSubstreams(n, T, new MRG32k3a(), statRuns);
		statRuns.quickSort();
		return (n - statRuns.getDoubleArrayList().lastIndexOf(0.0) - 1.0) / n;
	}
	
	

	// Initial value of X.
	public void initialState() {
		step = 0;
		X = X0;		
	}
	
	double max( double a, double b){
		double max;
		if(a<b)
			max= b;
		else
			max= a;
		return max;
	}
	
	int  maxindex(double [] a)
	   {  
		//double[] res= new double[2];
	  double  max = a[0];
	 // double max= 0;
	  int index =0;
	    for (int  i=0; i<a.length; i++){
	            
	                if (a[i] > max)	 {                   
	                        max = a[i];	 
	                        index= i;
	                }
	    } 	  
	    return index;
	} 
	
	int  indexprobability(double [] a)
	   {  
		int index = 0;
		double sum= 0;
		for (int i=0; i<a.length;i++)
			sum+=a[i];
		double p = Math.random();
		double cumulativeProbability = 0.0;
		for (int i=0; i<a.length;i++) {
		    cumulativeProbability += a[i];
		    if (p * sum <= cumulativeProbability ) {
		        index = i;
		    }
		}
	    return index;
	    
	    
	} 
	private  double minValue(double[] a) {
	    double min = a[0];
	    for (int ktr = 0; ktr < a.length; ktr++) {
	        if (a[ktr] < min) {
	            min = a[ktr];
	        }
	    }
	    return min;
	}
 boolean isProposedNegative()
	{
		double minValue = minValue(Xprop);
		if (minValue < 0)
			return true;
		else
			return false;
	}

	private void CalculatePropensities() {
	    a_sum = 0;		
	    a[0]=c[0]*X[0]*X[1];
		a[1]=c[1]*X[2];
		a[2]=c[2]*X[2];
		a[3]=c[3]*X[3];
		a[4]=c[4]*X[4]*X[5];
		a[5]=c[5]*X[6];
		a[6]=c[6]*X[6];
		a[7]=c[7]*X[7]*X[8];
		a[8]=c[8]*X[9];
		a[9]=c[9]*X[9];
		a[10]=c[10]*X[10];
		a[11]=c[11]*X[1]*X[8];
		a[12]=c[12]*X[11];
		a[13]=c[13]*X[11];
		for (int i=0; i<a.length; i++) {
			
			a_sum += a[i];
		}
	}
	
	
	
	protected int getV(int species, int reaction) {
		return v[species][reaction];
	}
	
	
	private static int[] remove(int[] a, int index) {
	    if (a == null || index < 0 || index >= a.length) {
	        return a;
	    }

	    int[] result = new int[a.length - 1];
	    for (int i = 0; i < index; i++) {
	        result[i] = a[i];
	    }

	    for (int i = index; i < a.length - 1; i++) {
	        result[i] = a[i + 1];
	    }

	    return result;
	}
	
	double min( double a, double b){
		double min;
		if(a < b)
			min= a;
		else
			min= b;
		return min;
	}
	
	
	/*private int identifyTheOnlyCriticalReaction(BitVector criticals, RandomStream steram) {
		double a_critical_sum = 0;
		for (int j=0; j<criticals.size(); j++) 
			if (criticals.get(j))
				a_critical_sum+=a[j];
		
		double r2 = UniformDist.inverseF(0, 1, steram.nextDouble());
		double test = r2*a_critical_sum;
		
		double sum = 0;
		for (int i=0; i<criticals.size(); i++) {
			if (!criticals.get(i)) continue;
			sum+=a[i];
			if (sum>=test) return i;
		}
		
		throw new RuntimeException("Drawing variable aborted!");
	}
*/

/*double[] computevariance(){
	ArrayList<Integer> ncr=  listOfCriticalReactions( );
	double [] var = new double[N];
	
	for (int i=0; i<N; i++){
		double sum=0;
	    for (int j=0; j<ncr.size();j++)
		sum+= Math.pow(getV(i, ncr.get(j)), 2)*a[ncr.get(j)];
		var[i]= sum;
	}
	return var;
}

double[] computemean(){
	ArrayList<Integer> ncr=  listOfCriticalReactions( );
	double [] var = new double[N];
	
	for (int i=0; i<N; i++){
		double sum=0;
	    for (int j=0; j<ncr.size();j++)
		sum+= getV(i, ncr.get(j))*a[ncr.get(j)];
		var[i]= sum;
	}
	return var;
}
void computeChanges()
{

	int is, js, ic;
	int changesNumber;
	
	// allocate space for the changes vectors

	
	for (int r=0; r< K; r++){
		changes = new int [K][Reactant[r].length + Product[r].length];
		nuChanges = new int [K][Reactant[r].length + Product[r].length];
		
	for ( int i = 0; i < (Reactant[r].length + Product[r].length); ++i)
	{
		changes[r][i] = 0;
		nuChanges[r][i] = 0;
	}

	
	
	// Merging the nu reactants and nu products into a 
	// Changes array
	
	for (is = 0; is < Reactant[r].length; ++is)
	{
		changes[r][is] = Reactant[r][is];
		nuChanges[r][is] = -nuReactant[r][is];
		//changes.push_back(reactants[is]);
		//nuChanges.push_back( -nuReactants[is] );
 	}

	
	//ic = reactants.size()+1;
	ic = Reactant[r].length;
	is = 0; 
	js = 0;
	while ( (is < Reactant[r].length) && (js < Product[r].length) )
	{
		if (changes[r][is] > Product[r][js])
		{
			changes [r][ic] = Product[r][js];
			nuChanges[r][ic] = nuProduct[r][js];
			++ic;
			++js;
		}
		else if (changes[r][is] < Product[r][js])
		{
			++is;
		}
		else
		{
			nuChanges[r][is] += nuProduct[r][js];
			++is;
			++js;
		}
	}
	for (; js < Product[r].length; ++js)
	{
		changes[r][ic] = Product[r][js];
		nuChanges[r][ic] = nuProduct[r][js];
		++ic;
	}
	changesNumber = ic;
	//cout << "ic: " << ic << endl;
	// remove the zero elements of the changes and nuChanges vectors
	is = 0;
	//while (is < changesNumber)
	while (is < changes[r].length)
	{
		if (nuChanges[r][is] == 0)
			
		{
			nuChanges[r] = remove(nuChanges[r], is);
			changes[r] = remove(changes[r], is);
			
			changesNumber -= 1;
			//cout << "is: " << is << endl;
		}
		else
		{
			++is;
		}
	}
	}
	
	for (int species=0; species<changes.length; species++) {
		for (int k=0; k<K; k++) {
			System.out.print(changes[species][k]);
		}
		System.out.println("");
	}
}*/



 

 /*void computeMuHatSigmaHat2()
{
	int is, ir, ns, indx, nr;
	double tmpfloat;
	nr = K;

	for (int numbS = 0; numbS < N; ++numbS)
	{
		mu[numbS] = 0.0;
		sigma[numbS] = 0.0;
	}

	for (ir = 0; ir < nr; ++ir)
	{
		
		double  riPropensity = a[ir];

		ns = changes[ir].length;
		for (is = 0; is < ns; is++ )
		{
			indx = changes[ir][is];
			tmpfloat = nuChanges[ir][is] * riPropensity;
			mu[indx] += tmpfloat;
			sigma[indx] += nuChanges[ir][is] * tmpfloat;
		}
	}

	// cout << "mu:"<<muHat << " sigma :"<<sigmaHat2 << endl;
}
 */
 
 
 
 /* void fireReactionProposed(int reactionIndex, double numberOfTimes)
	{
		
		int [] change = changes[reactionIndex];
		int [] nuChange =  nuChanges[reactionIndex];


		for (int i = 0; i < change.length; ++i)
		{
			Xprop[change[i]] += (nuChange[i]*((numberOfTimes)));
		}

		
	}
  
  
  void computePropensitiesGrowingVolume( double time, double genTime)
	{
		double volume	= 1. + time/genTime;
		double ivolume	= 1./volume;

		int nu;
		double x;
		double num, denom;

        double p;
		for (int ir = 0; ir < K; ++ir)
		{
			int order = 0;
			for (int i = 0; i < Reactant[ir].length; ++i)
			{
				order += nuReactant[ir][i];
			}

			p = c[ir];

			for (int s = 0; s < Reactant[ir].length; ++s)
			{
				nu		= nuReactant[ir][s];
				x		= X[ Reactant[ir][s] ];
				num		= x;
				denom	= nu;
				while ((--nu)>0)
				{
					denom	*= nu;
					num		*= (x - nu);
				}
				p *= ((double)num/(double)denom) ;
			}

			if (order == 2)
				p *=  ivolume ;

			if (order == 3)
				p *=    ivolume *ivolume;

			if (order > 3)
			{
				System.out.println("Aborting: Growing volume of reaction enviroment do not support reaction of order higher than 3, if you want it implement it");
				continue;
			}

			a[ir] = p;
		}
	}


	public double computeTimeStep( )
	{
		double HUGE_VAL = Double.MAX_VALUE;
	    double[] muHat = new double [N] ;
	    double[] varHat = new double [N] ;
	    computeHor();
	    Arrays.fill(muHat, 0.0);
	    varHat=  computevariance();
	    muHat=  computemean();
	    computeChanges();
	    computeHor();
	    computeMuHatSigmaHat2();

	    double tau, taup,  epsi, epsixi, epsixisq;
	    double xi;

	    tau = HUGE_VAL;

	    double a0 = 0;
	    for (int i=0; i<a.length; i++)
	    	a0+= a[i];
	    for (int is = 0; is < N; is++)
	    {
	        varHat[is] = sigma[is] - (1.0/a0) * muHat[is] * muHat[is];
	    }
	    

	    for (int is = 0; is < N; ++is)
	    {
	        taup = (HUGE_VAL*0.5);
	        xi = X[is];
	        switch (hor[is]) {
	            case 0:
	                break;
	            case 1:
	                epsi = epsilon;
	                epsixi = epsi * xi;
	                epsixi = max(epsixi,1.0);
	                tau = min(tau,epsixi/Math.abs(muHat[is]));
	                epsixisq = epsixi*epsixi;
	                tau = min(tau,epsixisq/varHat[is]);
	                break;
	            case 2:
	                if (nuHor[is] == 1)
	                    epsi = 0.5*epsilon;
	                else
	                    epsi = epsilon*(xi-1.0)/(2.0*(xi-1.0)+1.0);
	                epsixi = epsi * xi;
	                epsixi = max(epsixi,1.0);
	                tau = min(tau,epsixi/Math.abs(muHat[is]));
	                epsixisq = epsixi*epsixi;
	                tau = min(tau,epsixisq/varHat[is]);
	                break;
	            case 3:
	                if (nuHor[is]==1)
	                    epsi = 0.3333333333*epsilon;
	                else if (nuHor[is] == 2)
	                    epsi = epsilon*(xi-1)/(3.0*(xi-1)+1.5);
	                else
	                    epsi = epsilon*(xi-1)*(xi-2)/(3.0*(xi-1)*(xi-2)+(xi-2)+2.0*(xi-1));
	                epsixi = epsi * xi;
	                epsixi = max(epsixi,1.0);
	                tau = min(tau,epsixi/Math.abs(muHat[is]));
	                epsixisq = epsixi*epsixi;
	                tau = min(tau,epsixisq/varHat[is]);
	                break;
	            default:
	                break;
	        }
	    }

	    return tau;
	}*/
 /*void acceptNewSpeciesValues()
	{
		// XXX
		Xold = X;

		X = Xprop;
		Xprop = X;
	}
	 void reloadProposedSpeciesValues()
	{
		Xprop = X;
	}*/
	
	
	
	 
	 ///desactiver par la suite
	/*public void nextStep(RandomStream stream) {
		//int [] [] G = new double[K][N];
		double dt =0.0;
		 double genTime                  = 2100;
		double[] p = new double[K] ;
		X[1]  = 35 * (1 + step/genTime); //gennor(35   * (1 + t/genTime), 3.5);
        X[9]  = 350 * (1 + step/genTime); 
		 computePropensities();
       // computePropensitiesGrowingVolume( step,  genTime);
		 if (isNegative == false)
		 dt = computeTimeStep();	
		 for ( int k=0; k<K; k++){
			//if ( a[k] * dt>0) {
			  p[k] =   PoissonDist.inverseF(a[k] * dt,stream.nextDouble()) ;
			 // fireReactionProposed( k , p[k] );
			//}}
		 }
			  G=transposeMatrix(v);
				
				double [] sum;
				double [] res;
				double [] [] ress= new double[K][N];
				for ( int k=0; k<K; k++){
					 ress[k]=mutvc(G[k],p[k]);						 
				}
				double [] r = ress[0];
				for ( int k=1; k<K; k++){
				r=sumvv( r, ress[k]);
				
				}
				
				Xr  = sumvv(X,r); 
				//System.out.println("Xr0"+Xr[0]);
				X = Xr;
			
		 if (isProposedNegative() == false)
         {
             acceptNewSpeciesValues();
             
             t_old = t;
             t += dt;
           
             step ++;
             isNegative = false;
             
         }
         else
         {
            
             dt = dt * 0.5;
            // reloadProposedSpeciesValues();
             X = Xr;
             isNegative = true;
         }
		 
		 G=transposeMatrix(S);
			
			double [] sum;
			double [] res;
			double [] [] ress= new double[K][N];
			for ( int k=0; k<K; k++){
				 ress[k]=mutvc(G[k],p[k]);						 
			}
			double [] r = ress[0];
			for ( int k=1; k<K; k++){
			r=sumvv( r, ress[k]);
			
			}
			
			Xr  = sumvv(X,r); 
			//System.out.println("Xr0"+Xr[0]);
			X = Xr;
		
	}*/

	// Simulates the next step.
	/*public void nextStep(RandomStream stream) {
		//step+=tau;
		//step++;
		double tau1, tau2;
		int  index;
		boolean isNegative= false;
		double [] [] G = new double[K][N];
		double[] p = new double[K] ;
		a[0]=c[0]*X[0]*X[1];
		a[1]=c[1]*X[2];
		a[2]=c[2]*X[2];
		a[3]=c[3]*X[3];
		a[4]=c[4]*X[4]*X[6];
		a[5]=c[5]*X[7];
		a[6]=c[6]*X[7];
		a[7]=c[7]*X[4];
		a[8]=c[8]*X[8];
		a[9]=c[9]*X[10];
		
		System.out.println("step"+step);
		
		for ( int k=0; k<K; k++){
			
		  p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()) ;
			
		}
		

		
		
		
	
		
		tau1 = computeTau();	

		loop:{
ArrayList<Integer> ncr=  listOfCriticalReactions( );
double[]  pr = new double[ncr.size()];
System.out.println("size"+ncr.size());
		double a_crit=0;
	    for (int j=0; j<ncr.size();j++)
		a_crit+= a[ncr.get(j)];
		//tau2 = 		1/a_crit;	
	tau2= ExponentialDist.inverseF(1/a_crit,stream.nextDouble());
	//tau2= 1/(a_crit*stream.nextDouble());
if (tau1 < tau2){
		tau= tau1;
	
		// System.out.println("here");
for ( int k=0; k<K; k++){
for (int j=0; j<ncr.size();j++)
	if ( k== ncr.get(j))
		p[k] = 0;
	else 
        p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()) ;			
 }
 }
 else{
	// System.out.println("here2");
	 tau = tau2;	
	// double[]  pr = new double[ncr.size()];
	
	 for (int j=0; j<ncr.size();j++){
		 pr[j]= a[ncr.get(j)]/a_crit;
		// System.out.println("probability"+pr[j]);
	 }
	 index = indexprobability(pr);
	// System.out.println("index"+index);
	 //p[index] = 1;
	 //for (int j=0; j<ncr.size() && j!=index;j++)
	 for ( int k=0; k<K; k++){			
				if( k== index)
					p[k] = 1;
				else{
				    for (int j=0; j<ncr.size();j++)
				      if ( k== ncr.get(j) )
					     p[k] = 0;
				      else 
			             p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()) ;			
	            }			
 }}

 for ( int i=0; i<N; i++)
	 if (X[i]<0){
		 tau1= tau/2;
		 break loop;
	 }
	
	}
		
		G=transposeMatrix(S);
		
		double [] sum;
		double [] res;
		double [] [] ress= new double[K][N];
		for ( int k=0; k<K; k++){
			 ress[k]=mutvc(G[k],p[k]);						 
		}
		double [] r = ress[0];
		for ( int k=1; k<K; k++){
		r=sumvv( r, ress[k]);
		
		}
		
		Xr  = sumvv(X,r); 
		//System.out.println("Xr0"+Xr[0]);
		X = Xr;
 
		for ( int i=0; i<N; i++)
		 if (X[i]<0){
			 tau1= tau/2;
			 break loop;
		 }
		 else
			 step++;
		     X = Xr;
		 
		
		}
	 
									
	}*/
	
	/*private boolean leapBy(double tau, BitVector criticals, RandomStream stream) {
		int max = 0;
		int sum = 0;
		double [] [] ress= null;
		try {
			int times;
			double at;
			double [] res;
		
			int t= 0;
			for (int i=0; i<K; i++) {
				if (criticals.get(i)) continue;
				at = a[i]*tau;
				if (at>langevinThreshold) 
					times = Math.max(0, (int) Math.round(at+Math.sqrt(at)*NormalDist.inverseF01(stream.nextDouble())));
				else 
					times = PoissonDist.inverseF(at, stream.nextDouble());
				if (times>0){												
						 ress[t]=mutvc(G[i],times);	
						 t++;
					}
					
				
				max = Math.max(max, times);
				sum+=times;
			}
		double [] r = ress[0];
		for ( int s=1; s<ress.length; s++){
		r=sumvv( r, ress[s]);
		Xr  = sumvv(X,r); 
		X = Xr;
		}
		
			
			step++;
			return true;
		} catch (RuntimeException e) {
			return false;
		}
	}*/
	/*public void nextStep(RandomStream stream) {
		//step+=tau;
		//step++;
		double tau1 = 0, tau2;
		int  index;
		boolean isNegative= false;
		//double [] [] G = new double[K][N];
		G=transposeMatrix(v);
		double[] p = new double[K] ;
		//CalculatePropensities();
		a =calculatePropensity( X);
		
	//	System.out.println("step"+step);
		//computePropensities();
	
		identifyCriticals();
		 if (isNegative == false)
		tau1 = chooseTauNonCriticals(criticals);
		//tau2 = chooseTauCriticals(criticals,stream) ;
		
		

		
		
			
		
			if (tau1<useSimpleFactor/a_sum) {
				
				
				performSSA(100, stream);
				//System.out.println("Call SSA");
				
			}
			else { 
				tau2 = chooseTauCriticals(criticals, stream);

				if (tau1 < tau2) {
					System.out.println("tau1");
					tau = tau1;
					
					for ( int k=0; k<K; k++){
						if(criticals.get(k))
								p[k] = 0;
							else 
						        p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()) ;			
				    }
						 
					
					
				}
				else{
					System.out.println("tau2");
					tau = tau2;
					double[]  pr = new double[criticals.size()];
					for (int j=0; j<criticals.size();j++){
						if (criticals.get(j))
						 pr[j]= a[j]/a_sum_c;
						// System.out.println("probability"+pr[j]);
					 }
					
					 index = indexprobability(pr);
						// System.out.println("index"+index);
						 //p[index] = 1;
						 //for (int j=0; j<ncr.size() && j!=index;j++)
						 for ( int k=0; k<K; k++){			
									if( k== index)
										p[k] = 1;
									else{
									    
									      if (  criticals.get(k) )
										     p[k] = 0;
									      else 
								             p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()) ;			
						            }			
					 }
						 }
				}
			
			if (isProposedNegative() == false){
				
				
				G=transposeMatrix(v);
				
				double [] sum;
				double [] res;
				double [] [] ress= new double[K][N];
				for ( int k=0; k<K; k++){
					 ress[k]=mutvc(G[k],p[k]);						 
				}
				double [] r = ress[0];
				for ( int k=1; k<K; k++){
				r=sumvv( r, ress[k]);
				
				}
				
				Xr  = sumvv(X,r); 
				//System.out.println("Xr0"+Xr[0]);
				X = Xr;
				step++;
				isNegative =false;
			}
			else{
					
					tau/=2.0;
					isNegative =true;
					//X = Xr;
				}
			}*/
	
	
	public void nextStep(RandomStream stream) {
		step++;
		double[] p = new double[K] ;
		double genTime = 2100; 	
     
		//double [] [] G = new double[K][N];
		
		
		//CalculatePropensities();
		
		a =calculatePropensity( X);
		
		for ( int k=0; k<K; k++){
			
		  p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()) ;
		 
		}
		

		
		
		
		
		
		G=transposeMatrix(v);
		
		double [] sum;
		double [] res;
		double [] [] ress= new double[K][N];
		for ( int k=0; k<K; k++){
			 ress[k]=mutvc(G[k],p[k]);
			
			 
		}
		double [] r = ress[0];
		for ( int k=1; k<K; k++){
		r=sumvv( r, ress[k]);
		
		}
		
		Xr  = sumvv(X,r); 
		X= Xr;
			}
	
	/*public void nextStep(RandomStream stream) {
		//step+=tau;
		step++;
		double[] p = new double[K] ;
		a =calculatePropensity( X);
		a[0]=c[0]*X[0]*X[1];
		a[1]=c[1]*X[2];
		a[2]=c[2]*X[2];
		a[3]=c[3]*X[3];
		a[4]=c[4]*X[4]*X[5];
		a[5]=c[5]*X[6];
		a[6]=c[6]*X[6];
		a[7]=c[7]*X[7]*X[8];
		a[8]=c[8]*X[9];
		a[9]=c[9]*X[9];
		a[10]=c[10]*X[10];
		a[11]=c[11]*X[1]*X[8];
		a[12]=c[12]*X[11];
		a[13]=c[13]*X[11];
		for ( int k=0; k<K; k++){
			
		  p[k] =   PoissonDist.inverseF(a[k] * tau,stream.nextDouble()) ;
		 
		}		
		
		G=transposeMatrix(v);
		
		double [] sum;
		double [] res;
		double [] [] ress= new double[K][N];
		for ( int k=0; k<K; k++){
			 ress[k]=mutvc(G[k],p[k]);
			
			 
		}
		double [] r = ress[0];
		for ( int k=1; k<K; k++){
		r=sumvv( r, ress[k]);
		
		}
		
		Xr  = sumvv(X,r); 
		X= Xr;
		if(step==T)
	    	   R=X[0];
	
									
	}							*/		
	
	// Returns the statistics (performance measure) accumulated so far.
	// Valid only after last step (at time t).
	public double getPerformance() {		
		//return X[0]*X[0];
		return X[5];
	}
	public double[] getState() {	
		return X;
	}
	

	public int compareTo(MarkovChainComparable m, int i) {
		if (!(m instanceof cAMPComparable)) {
			throw new IllegalArgumentException("Can't compare a " + "Biology example with other types of Markov chains.");
		}
		double  mx;
		
			mx = ((cAMPComparable) m).X[i];
			return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));
		
	}	
	
	 // Returns the state transformed to (0,1)^N.
    public double[] getPoint() {
    	double[] state01 = new double[N];
    	for(int i=0;i<N;i++)
        state01[i] = getCoordinate(i);       
        return state01;
    }
		
 // Returns coordinate j of the state mapped to (0,1).
    public double getCoordinate (int j) {
     double zvalue;                 
 	/*  //zvalue = (X[j]- X0[j])/(Math.sqrt(2*a[j]*step*X0[j])) ;          
        // zvalue = (X[j]- X0[j])/(Math.sqrt(X0[j])) ;    
    	 zvalue = (X[j]- X0[j])/(Math.sqrt((c[0]+c[1]+c[2]+c[3])*step*X0[j])) ;
        return NormalDist.cdf01 (zvalue);*/
    	
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
    	
    	/*switch (j) {
        case 0:                
            return map.realTo01 (X[0]);
        case 1:               
            return map.realTo01 (X[1]);
        case 2:                
            return map.realTo01 (X[2]);
        default:
            throw new IllegalArgumentException("Invalid state index");
    	}*/
        
    }
   
    // Returns the state transformed to (0,1)^2.
    public int dimension() {
    	return N;
     }


	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append(" cAMP_cascade model :\n");
		//for (int n=0; n<N; n++){
		//sb.append(" Biology example:\n");
		//sb.append(" X() : number of molecules of types k                  = " + PrintfFormat.format(8, 3, 1, X[n]) + "\n");
		
		//}
		return sb.toString();
	}
}