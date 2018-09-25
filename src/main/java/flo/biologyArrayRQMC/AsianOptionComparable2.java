package flo.biologyArrayRQMC;

// Asian option based on a geometric Brownian motion, for array-RQMC experiments.
// Average is based on d equidistant observation times, first at t1=T/d and last at T. 
// Time is in years.
// The state contains:
// (1) state of geo. Brownian motion;  (2) the current average of values observed so far.

// State is transformed to (0,1)^2 as follows:
// (1) logS has a known lognormal distribution and we transform it to uniform 
//  by applying the cdf of this lognormal.
// (2) for the current average sumS/step, we assume (as an approximation) that it has 
//  the same lognormal distribution as the geometric average, and we apply again the cdf.
// Note: Even if the approx in (2) was exact, the transformed state would *not* be uniform 
// over [0,1)^2, because the two components are not independent.


import umontreal.ssj.rng.*;
import umontreal.ssj.markovchainrqmc.*;
import umontreal.ssj.util.sort.*;
import umontreal.ssj.util.PrintfFormat;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.stat.TallyStore;

class AsianOptionComparable2 extends MarkovChainComparable
      implements MultiDim01, MultiDim  {

    int d;                         // Number of observation times.
    double strike;                 // Strike price.
    double td;                     // Expiration date (in years)
    double t1;                     // First obs. time (in years)
    double s0;                     // Asset value at time t1.
    double r = Math.log(1.09);     // risk-free interest rate
    double discount;
    double sigma;                  // Volatility
    double sigma2;                 // sigma squared
    double delta;               // Time between observations. 
    double muDelta1;            // Trend for first step.
    double muDelta;             // Trend for each other step.
    double sigmaSqrtDelta1;     // Sqrt of time to first observation. 
    double sigmaSqrtDelta;      // sigma * sqrt{delta}
    double logS;            // State of Brownian motion, ln S(t).
    double logS0;           // Initial state of Brownian motion, ln S(0).
    double sumS;            // Current sum of asset values at observation points.
    double S;               // Current asset value.
    int step;               // Current step.

    // Arrays to store precomputed values used in the mapping to [0,1).
    double[] sigmaSqrtSumDelta;   // sigma * sqrt{t1 + (j-1)*delta}
    double[] meanGeoAv;           // Mean of log of geometric average up to each step.
    double[] stdevGeoAv;          // Variance of log of geometric average up to each step.


    public AsianOptionComparable2 (double r, int d, double t1, double T, double K, 
    		 double s0, double sigma) {
        this.r = r;
        this.d = d;
        this.t1 = t1;
        this.td = T;
        this.strike = K;
        this.s0 = s0;
        this.sigma = sigma;
        logS0 = Math.log(s0);
        delta = (td - t1) / (double)(d-1);
        discount = Math.exp(-r * td);
        muDelta1 = (r - 0.5 * sigma * sigma) * t1;
        muDelta  = (r - 0.5 * sigma * sigma) * delta;
        sigmaSqrtDelta1 = sigma * Math.sqrt(t1);
        sigmaSqrtDelta  = sigma * Math.sqrt(delta);
        stateDim = 2;

        sigmaSqrtSumDelta = new double[d+1];
        meanGeoAv = new double[d+1];
        stdevGeoAv = new double[d+1];
        double sumSqj = 0.0;
        for (int j=1; j<=d; j++) {
        	sigmaSqrtSumDelta[j]  = sigma * Math.sqrt(t1 + (j-1)*delta);
        	meanGeoAv[j] = logS0 + muDelta1 + muDelta * 0.5 * j * (j-1);
        	sumSqj += (j-1)^2;
        	stdevGeoAv[j] = sigma * Math.sqrt ((t1 + delta * sumSqj / (j*j)));
        }
    }

    public double calcFractionPositivePayoff (int n) {
        TallyStore statRuns = new TallyStore();
        simulRunsWithSubstreams(n, d, new MRG32k3a(), statRuns);
        statRuns.quickSort();
        return (n - statRuns.getDoubleArrayList().lastIndexOf(0.0) - 1.0) / n;
    }

    // Initial value of underlying Brownian motion is zero.
    public void initialState() {
        step = 0;
        sumS = 0.0;
    }

    // Simulates the next step.
    public void nextStep(RandomStream stream) {
        step++;
        double z = NormalDist.inverseF01(stream.nextDouble());
        if (step == 1) {
        	logS = logS0 + muDelta1 + sigmaSqrtDelta1 * z;
        } else {
        	logS += muDelta + sigmaSqrtDelta * z;
        }
        S = Math.exp(logS);
        sumS += S;
    }
    
   

    // Returns the net payoff (valid after the last step, at time T).
    public double getPerformance() {
        double value = ((sumS / (double) d) - strike) * discount;
        if (value <= 0.0) {
            value = 0.0;
        }
        return value;
    }

    // Compare this chain to other chain based on criterion j:
    // S if j=0 and sumS if j=1.
    // We assume that both chains are at the same step.  
    public int compareTo(MarkovChainComparable other, int j) {
        if (!(other instanceof AsianOptionComparable2)) {
            throw new IllegalArgumentException("Can't compare a "
                    + "AsianOption with other types of Markov chains.");
        }
        double x, mx;
        switch (j) {
            case 0:
                mx = ((AsianOptionComparable2) other).S;
                return (S > mx ? 1 : (S < mx ? -1 : 0));
            case 1:
                x = sumS;
                mx = ((AsianOptionComparable2) other).sumS;
                return (x > mx ? 1 : (x < mx ? -1 : 0));
            default:
                throw new IllegalArgumentException("Invalid state index");
        }
    }
    

    // Returns the state transformed to (0,1)^2.
    public double[] getPoint() {
    	double[] state01 = new double[2];
        state01[0] = getCoordinate(0);
        state01[1] = getCoordinate(1);
        return state01;
    }

    // Returns coordinate j of the state mapped to (0,1).
    public double getCoordinate (int j) {
        double zvalue;
        switch (j) {
            case 0:
                zvalue = logS - logS0 - muDelta1 - ((step-1) * muDelta);
                zvalue /= sigmaSqrtSumDelta[step];
                return NormalDist.cdf01 (zvalue);
            case 1:
                zvalue = Math.log(sumS/step) - meanGeoAv[step];
                zvalue /= stdevGeoAv[step];
                return NormalDist.cdf01 (zvalue);
            default:
                throw new IllegalArgumentException("Invalid state index");
        }
    }
   
    // Returns the state transformed to (0,1)^2.
    public int dimension() {
    	return 2;
     }
    public double[] getState() {
		double[] state ={S,sumS};
		return state;
	}
    

    

    public String toString() {
        StringBuffer sb = new StringBuffer("----------------------------------------------\n");
        sb.append("Pricing of an Asian option:\n");
        sb.append(" S(0)   =                 = "
                + PrintfFormat.format(8, 3, 1, s0) + "\n");
        sb.append(" d      =                 = "
                + PrintfFormat.format(8, 3, 1, d) + "\n");
        sb.append(" delta  =                 = "
                + PrintfFormat.format(8, 3, 1, delta) + "\n");
        sb.append(" start date t1    =          = "
                + PrintfFormat.format(8, 3, 1, t1) + "\n");
        sb.append(" time horizon T  =          = "
                + PrintfFormat.format(8, 3, 1, td) + "\n");
        sb.append(" interest rate r  =        = "
                + PrintfFormat.format(8, 3, 1, r) + "\n");
        sb.append(" volatility sigma =             = "
                + PrintfFormat.format(8, 3, 1, sigma) + "\n");
        sb.append(" strike price K   =         = "
                + PrintfFormat.format(8, 3, 1, strike) + "\n");
        return sb.toString();
    }
}
