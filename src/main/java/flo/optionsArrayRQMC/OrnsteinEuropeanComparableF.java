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

class OrnsteinEuropeanComparableF extends MarkovChainComparable
      implements MultiDim01, MultiDim{


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
      double alpha;
      double b,correlation;
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
    public OrnsteinEuropeanComparableF (double r, double sigma,double T,int d, double alpha, double b,double S0,double V0,double K, double correlation,RealsTo01Map map) {    	  
        this.r = r;
        this.sigma = sigma;
        this.alpha = alpha;
        this.K = K;
        this.T = T;
        this.d=d;
        this.correlation = correlation;
        this.b = b;
        this.h=(double)T/d;
        z = new double[2];
        stateDim=2;
        this.map = map;
        //Model = new HestonVolatilityModel(r,sigma,lambda,xi);
        //initialStat = new State(S0, V0);
        this.S0= S0;
        this.V0=V0;
        normalGen= new MultinormalCholeskyGen(
                new NormalGen(null),
                new double[]{0.0, 0.0},
                new double[][]{{1.0, correlation}, {correlation, 1.0}});
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
        Vtt =  Vt + h * alpha *(b-Vt) + sigma *  Math.sqrt(h ) * z[1];    	     
        Stt= (1.0+ r * h ) * St  + Math.sqrt(h ) *Math.exp(Vt )   * z[0];
        Vt=Vtt;
        St=Stt;       
        sumS += St;       
        if(step==d)
     	   ST=St;
//       initialStat=state;
       //Vt=Vtt;
       
       
 /*      step++;
       
       normalGen= new MultinormalCholeskyGen(
               new NormalGen(null),
               new double[]{0.0, 0.0},
               new double[][]{{1.0, xi}, {xi, 1.0}});
       
       normalGen.setStream(new MRG32k3a());
       normalGen.nextPoint(z);
//       state =Model.advanceEuler(initialStat,z, h*step);
    	Vt = sigma * sigma + Math.exp(-lambda * h*step) * (
//                state.getVolatility()
                Vt- sigma * sigma
                + xi * Math.sqrt(h*step * Vt) * z[1]
                );
    	
    	Vt = Math.max(0, Vt);

        St= (
              1.0
              + r * h*step
//              + Math.sqrt(h*step * state.getVolatility()) * z[0]
              + Math.sqrt(h*step * Vt) * z[0]
              ) * St;
//       Vt=state.getVolatility();
//       St=state.getPrice();       
       sumS += St;
       initialStat=state;*/
    }
    
 /*   public void nextStep(RandomStream stream) {
        step++;
       
        normalGen= new MultinormalCholeskyGen(
                new NormalGen(null),
                new double[]{0.0, 0.0},
                new double[][]{{1.0, xi}, {xi, 1.0}});
        
        normalGen.setStream(new MRG32k3a());
        normalGen.nextPoint(z);
        state =Model.advanceEuler(initialStat,z, h*step);
//     	Vt = sigma * sigma + Math.exp(-lambda * h*step) * (
//                 state.getVolatility()
//                 - sigma * sigma
//                 + xi * Math.sqrt(h*step * state.getVolatility()) * z[1]
//                 );
//     	
//     	Vt = Math.max(0, Vt);
 //
//         St= (
//               1.0
//               + r * h*step
//               + Math.sqrt(h*step * state.getVolatility()) * z[0]
//               ) * state.getPrice();
        Vt=state.getVolatility();
        St=state.getPrice();       
        sumS += St;
        initialStat=state;
        
        
        step++;
        
        normalGen= new MultinormalCholeskyGen(
                new NormalGen(null),
                new double[]{0.0, 0.0},
                new double[][]{{1.0, correlation}, {correlation, 1.0}});
        
        normalGen.setStream(new MRG32k3a());
        normalGen.nextPoint(z);
//        state =Model.advanceEuler(initialStat,z, h*step);
     	Vt = sigma * sigma + Math.exp(-lambda * h*step) * (
//                 state.getVolatility()
                 Vt- sigma * sigma
                 + xi * Math.sqrt(h*step * Vt) * z[1]
                 );
     	
     	Vt = Math.max(0, Vt);

         St= (
               1.0
               + r * h*step
//               + Math.sqrt(h*step * state.getVolatility()) * z[0]
               + Math.sqrt(h*step * Vt) * z[0]
               ) * St;
//        Vt=state.getVolatility();
//        St=state.getPrice();       
        sumS += St;
        initialStat=state;
     }*/
//  state= Model.advanceEuler(initialState, z, step*h);
//  St=state.
//  sumS += St;

    // Returns the net payoff (valid after the last step, at time T).
    /*public double getPerformance() {
        double value = ((sumS -S0/2-ST/2)* h- K) * Math.exp(-r * T);
        if (value <= 0.0) {
            value = 0.0;
        }
        return value;
    }*/
    public double getPerformance() {
        double value = (ST - K) * Math.exp(-r * T);
        if (value <= 0.0) {
            value = 0.0;
        }
        return value;
    }
    // Compare this chain to other chain based on criterion j:
    // S if j=0 and sumS if j=1.
    // We assume that both chains are at the same step.
   


    // Returns the state transformed to (0,1)^2.
    public double[] getPoint() {
    	double[] state01 = new double[2];
        state01[0] = getCoordinate(0);
        state01[1] = getCoordinate(1);
       // state01[2] = getCoordinate(2);
        return state01;
    }

  
    public double getCoordinate (int j) {
        switch (j) {
            case 0:                
                return map.realTo01 (Vt);
            case 1:               
                return map.realTo01 (St);
          //  case 2:                
               // return map.realTo01 (sumS);
            default:
                throw new IllegalArgumentException("Invalid state index");
        }
    }

    public int compareTo(MarkovChainComparable other, int j) {
        if (!(other instanceof OrnsteinEuropeanComparableF)) {
            throw new IllegalArgumentException("Can't compare a "
                    + "AsianOption with other types of Markov chains.");
        }
        double x, mx;
        switch (j) {
            case 0:
            	  mx = ((OrnsteinEuropeanComparableF) other).Vt;
            	  return (Vt > mx ? 1 : (Vt < mx ? -1 : 0));
            /*case 1:
                mx = ((HestonModelEuropeanComparable) other).sumS;
                return (sumS > mx ? 1 : (sumS < mx ? -1 : 0));*/
            case 1:
            	 mx = ((OrnsteinEuropeanComparableF) other).St;
                 return (St > mx ? 1 : (St < mx ? -1 : 0));
              
              
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
	public int dimension() {
		// TODO Auto-generated method stub
		return 2;
	}

	@Override
	public double[] getState() {
		
		double [] state= {St,Vt};
		return state;
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


