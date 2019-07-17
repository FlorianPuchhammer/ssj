package flo.optionsArrayRQMC;

import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.randvar.NormalGen;
import umontreal.ssj.randvarmulti.MultinormalCholeskyGen;
import umontreal.ssj.randvarmulti.MultinormalGen;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.TallyStore;

import umontreal.ssj.util.PrintfFormat;
import umontreal.ssj.util.sort.MultiDim;
import umontreal.ssj.util.sort.MultiDim01;

class HestonModelComparablef extends MarkovChainComparable
      implements  MultiDim01, MultiDim {


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


     // model parameters
      double r;
      double sigma;
      double lambda;
      double xi,correlation;
      MultinormalGen normalGen;
    
   int d;                         // Number of observation times.
   double S0;    
   double V0; 
   double sumS;            // Current sum of asset values at observation points.
//	double S;               // Current asset value.
   int step;               // Current step.
   RealsTo01Map map;       // One-dimensional map from reals to [0,1]
   double[] z;
   double h,St,Vt,K,T,Stt,Vtt,ST;
   // HestonVolatilityModel Model;
    //State initialStat;
    //State state;
    
//   NormalPairModel npm;
   double[] meanSt;   
   double[] stdSt; 
   double[] meanVt;   
   double[] stdVt; 
   double[] meansumSt;   
   double[] stdsumSt; 
   double[] meanGeoSt;           // Mean of log of geometric average up to each step.
   double[] stdevGeoSt;          // Variance of log of geometric average up to each step.
    public HestonModelComparablef (double r, double sigma,double T,int d, double lambda, double xi,double S0,double V0,double K, double correlation,RealsTo01Map map) {    	  
        this.numSteps = d;
    	this.r = r;
        this.sigma = sigma;
        this.lambda = lambda;
        this.K = K;
        this.T = T;
        this.d=d;
        this.correlation = correlation;
        this.xi = xi;
        this.h=(double)T/d;
        z = new double[2];
        stateDim=3;
        //Model = new HestonVolatilityModel(r,sigma,lambda,xi);
        //initialStat = new State(S0, V0);
        this.S0= S0;
        this.map =map;
        this.V0=V0;
        normalGen= new MultinormalCholeskyGen(
                new NormalGen(null),
                new double[]{0.0, 0.0},
                new double[][]{{1.0, correlation}, {correlation, 1.0}});
        meanSt = new double[d+1];
        stdSt = new double[d+1];
        meanVt = new double[d+1];
        stdVt = new double[d+1];
        meansumSt = new double[d+1];
        stdsumSt = new double[d+1];
        meanGeoSt = new double[d+1];
        stdevGeoSt = new double[d+1];
      
        //double sumSqj = 0.0;
        meanSt[0]=0.0;
        stdSt[0]=0.0;
        meanVt[0]=0.0;
        stdVt[0]=0.0;
        meanGeoSt[0]=0.0;
        stdevGeoSt[0]=0.0;
        meansumSt[0]=0.0;
        stdsumSt[0]=0.0;

        for (int j=1; j<=d; j++) {        	
        	//meanGeoSt[j] = Math.log(S0) + ((w+r) +muu * muu )* obsTimes[j]  ;
        	meanVt[j]=0.0;
            stdVt[j]=0.0;
        	meanGeoSt[j] = 0.0  ;
        	meansumSt[j] += meanGeoSt[j];
        	//double gam1 = gamma (obsTimes[j] * muu*muu/(double)nu +0.5);
       //double gam2 = gamma (obsTimes[j] * muu*muu/(double)nu );
        	//double gam = gam1 * gam1/(double)(gam2 * gam2);
        	//double tt = nu/(double)(obsTimes[j] *muu*muu);
        	//double rr =muu * muu* muu /(double) (nu * nu) ;
        	stdevGeoSt[j]= 0.0;
        	
        	stdsumSt[j] +=  0.0;
        	stdsumSt[j] = Math.sqrt (stdsumSt[j]);
        }
     }
    
    public double calcFractionPositivePayoff (int n) {
        TallyStore statRuns = new TallyStore();
        System.out.println("okkk");
        simulRunsWithSubstreams(n, d, new MRG32k3a(), statRuns);
        statRuns.quickSort();
        return (n - statRuns.getDoubleArrayList().lastIndexOf(0.0) - 1.0) / n;
    }

    // Initial value of underlying Brownian motion is zero.
    public void initialState() {
        step = 0;
        Vt = V0;
        St = S0;
        sumS = 0.0;
        
    }

    // Simulates the next step.
    public void nextStep(RandomStream stream) {
       step++;                   
       normalGen.setStream(stream);
       normalGen.nextPoint(z);
       //state =Model.advanceEuler(initialStat,z, h*step);
       Vtt = sigma * sigma + Math.exp(-lambda * h) * (Vt- sigma * sigma + xi * Math.sqrt(h * Vt) * z[1]);    	
       Vtt = Math.max(0, Vtt);
       Stt= (1.0+ r * h + Math.sqrt(h * Vt) * z[0]) * St;
       Vt=Vtt;
       St=Stt;       
       sumS += St;       
       if(step==d)
    	   ST=St;

    }
    
   
 
    // Returns the net payoff (valid after the last step, at time T).
   
    public double getPerformance() {
        double value = ((sumS +S0*0.5-ST*0.5)* h- K) * Math.exp(-r * T);
        if (value <= 0.0) {
            value = 0.0;
        }
        return value;
    }
/*    public double getPerformance() {
         double value = (ST- K) * Math.exp(-r * T);
        if (value <= 0.0) {
            value = 0.0;
        }
        return value;
    }*/
    // Compare this chain to other chain based on criterion j:
    // S if j=0 and sumS if j=1.
    // We assume that both chains are at the same step.
   


    // Returns the state transformed to (0,1)^2.
   /* public double[] getPoint() {
    	double[] state01 = new double[2];
        state01[0] = getCoordinate(0);
        state01[1] = getCoordinate(1);
        return state01;
    }*/
    
    public double[] getPoint() {
    	double[] state01 = new double[3];
        state01[0] = getCoordinate(0);
        state01[1] = getCoordinate(1);
        state01[2] = getCoordinate(2);
        return state01;
    }



    
    public double getCoordinate (int j) {
    	double zvalue;
        switch (j) {
            case 0:                
            	zvalue = St-meanVt[step];
                zvalue /= stdVt[step];
                return map.realTo01 (zvalue);
      
            case 1:                
            	zvalue = St-meanGeoSt[step];
                zvalue /= stdevGeoSt[step];
                return map.realTo01 (zvalue);
            case 2:               
            	zvalue = Math.log(sumS/step) - meansumSt[step];
                zvalue /= stdsumSt[step];
                return map.realTo01 (zvalue);
            default:
                throw new IllegalArgumentException("Invalid state index");
        }
    }
    public int compareTo(MarkovChainComparable other, int j) {
        if (!(other instanceof HestonModelComparablef)) {
            throw new IllegalArgumentException("Can't compare a "
                    + "AsianOption with other types of Markov chains.");
        }
        double  mx;
        switch (j) {
            case 0:
                mx = ((HestonModelComparablef) other).St;
                return (St > mx ? 1 : (St < mx ? -1 : 0));
           
            case 2:
                mx = ((HestonModelComparablef) other).Vt;
                return (Vt > mx ? 1 : (Vt < mx ? -1 : 0));   
            case 1:
                mx = ((HestonModelComparablef) other).sumS / (double) step;
                return (sumS / (double)step > mx ? 1 : (sumS / (double) step < mx ? -1 : 0));
            default:
                throw new IllegalArgumentException("Invalid state index");
        }
    }
    
    
   
    
    public String toString() {
        StringBuffer sb = new StringBuffer("----------------------------------------------\n");
        sb.append("Pricing of an Asian option:\n");
        sb.append(" d      =                 = "
                + PrintfFormat.format(8, 3, 1, d) + "\n");
 
        
        sb.append(" interest rate r  =        = "
                + PrintfFormat.format(8, 3, 1, r) + "\n");
        sb.append(" volatility sigma =             = "
                + PrintfFormat.format(8, 3, 1, sigma) + "\n");
        
        return sb.toString();
    }

	@Override
	public double[] getState() {
		double [] state = {St, sumS/(double)step,Vt};
		return state;
	}

	@Override
	public int dimension() {
		// TODO Auto-generated method stub
		return 3;
	}
}
//	@Override
//	public int compareTo(MarkovChainComparable other, int j) {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//	@Override
//	public double[] getPoint() {
//		// TODO Auto-generated method stub
//		return null;
//	}
//	@Override
//	public double getCoordinate(int j) {
//		// TODO Auto-generated method stub
//		return 0;
//	}
//	@Override
//	public void initialState() {
//		// TODO Auto-generated method stub
//		
//	}
//	@Override
//	public void nextStep(RandomStream stream) {
//		// TODO Auto-generated method stub
//		
//	}
//	@Override
//	public double getPerformance() {
//		// TODO Auto-generated method stub
//		return 0;
//	}}


