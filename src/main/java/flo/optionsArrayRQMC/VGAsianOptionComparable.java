package flo.optionsArrayRQMC;
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


import java.util.Arrays;

import umontreal.ssj.rng.*;
import umontreal.ssj.markovchainrqmc.*;
import umontreal.ssj.util.*;
import umontreal.ssj.util.sort.MultiDim;
import umontreal.ssj.util.sort.MultiDim01;
import umontreal.ssj.probdist.GammaDist;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.randvar.GammaGen;
import umontreal.ssj.stat.TallyStore;
import umontreal.ssj.stochprocess.BrownianMotion;
import umontreal.ssj.stochprocess.BrownianMotionBridge;
import umontreal.ssj.stochprocess.GammaProcess;
import umontreal.ssj.stochprocess.GammaProcessBridge;
import umontreal.ssj.stochprocess.GammaProcessSymmetricalBridge;
import umontreal.ssj.stochprocess.GeometricVarianceGammaProcess;
import umontreal.ssj.stochprocess.VarianceGammaProcess;
import umontreal.ssj.stochprocess.VarianceGammaProcessDiff;

class VGAsianOptionComparable extends MarkovChainComparable
      implements MultiDim01, MultiDim {


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
    private static double loggamma(double x) 
    { 
        int i; 
        double r, x2, y, f, u0, u1, u, z; 
        double[] b = new double[19]; 

        if (x > 13.0) 
        { 
            r = 1.0; 
            while (x <= 22.0) 
            { 
                r /= x; 
                x += 1.0; 
            } 
            x2 = -1.0 / (x * x); 
            r = Math.log(r); 
            return Math.log(x) * (x - 0.5) - x + r + 0.918938533204672 + 
                    (((0.595238095238095e-3 * x2 + 0.793650793650794e-3) * x2 + 
                    0.277777777777778e-2) * x2 + 0.833333333333333e-1) / x; 
        } 
        else 
        { 
            f = 1.0; 
            u0 = u1 = 0.0; 
            b[1] = -0.0761141616704358; b[2] = 0.0084323249659328; 
            b[3] = -0.0010794937263286; b[4] = 0.0001490074800369; 
            b[5] = -0.0000215123998886; b[6] = 0.0000031979329861; 
            b[7] = -0.0000004851693012; b[8] = 0.0000000747148782; 
            b[9] = -0.0000000116382967; b[10] = 0.0000000018294004; 
            b[11] = -0.0000000002896918; b[12] = 0.0000000000461570; 
            b[13] = -0.0000000000073928; b[14] = 0.0000000000011894; 
            b[15] = -0.0000000000001921; b[16] = 0.0000000000000311; 
            b[17] = -0.0000000000000051; b[18] = 0.0000000000000008; 
            if (x < 1.0) 
            { 
                f = 1.0 / x; 
                x += 1.0; 
            } 
            else 
                while (x > 2.0) 
                { 
                    x -= 1.0; 
                    f *= x; 
                } 
            f = Math.log(f); 
            y = x + x - 3.0; 
            z = y + y; 
            for (i = 18; i >= 1; i--) 
            { 
                u = u0; 
                u0 = z * u0 + b[i] - u1; 
                u1 = u; 
            } 
            return (u0 * y + 0.491415393029387 - u1) * (x - 1.0) * (x - 2.0) + f; 
        } 
    } 
	private static double recipgamma(double x, double odd, double even) 
    { 
        int i; 
        double alfa, nu, x2; 
        double[] b = new double[13]; 

        b[1] = -0.283876542276024; b[2] = -0.076852840844786; 
        b[3] = 0.001706305071096; b[4] = 0.001271927136655; 
        b[5] = 0.000076309597586; b[6] = -0.000004971736704; 
        b[7] = -0.000000865920800; b[8] = -0.000000033126120; 
        b[9] = 0.000000001745136; b[10] = 0.000000000242310; 
        b[11] = 0.000000000009161; b[12] = -0.000000000000170; 
        x2 = x * x * 8.0; 
        alfa = -0.000000000000001; 
        nu = 0.0; 
        for (i = 12; i >= 2; i -= 2) 
        { 
            nu = -(alfa * 2.0 + nu); 
            alfa = -nu * x2 - alfa + b[i]; 
        } 
        even = (nu / 2.0 + alfa) * x2 - alfa + 0.921870293650453; 
        alfa = -0.000000000000034; 
        nu = 0.0; 
        for (i = 11; i >= 1; i -= 2) 
        { 
            nu = -(alfa * 2.0 + nu); 
            alfa = -nu * x2 - alfa + b[i]; 
        } 
        odd = (alfa + nu) * 2.0; 
        return odd * x + even; 
    } 
    
    private static double gamma(double x) 
    { 
        int inv; 
        double y, s, f = 0.0, g, odd = 0.0, even = 0.0; 

        if (x < 0.5) 
        { 
            y = x - Math.floor(x / 2.0) * 2; 
            s = Math.PI; 
            if (y >= 1.0) 
            { 
                s = -s; 
                y = 2.0 - y; 
            } 
            if (y >= 0.5) y = 1.0 - y; 
            inv = 1; 
            x = 1.0 - x; 
            f = s / Math.sin(3.14159265358979 * y); 
        } 
        else 
            inv = 0; 
        if (x > 22.0) 
            g = Math.exp(loggamma(x)); 
        else 
        { 
            s = 1.0; 
            while (x > 1.5) 
            { 
                x = x - 1.0; 
                s *= x; 
            } 
            g = s / recipgamma(1.0 - x, odd, even); 
        } 
        return (inv == 1 ? f / g : g); 
    } 


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
    double S0;           // Initial state of Brownian motion, ln S(0).
    double sumS, logsumS;            // Current sum of asset values at observation points.
    double w;               // Current asset value.
    int step, substep;               // Current step.
    RealsTo01Map map;       // One-dimensional map from reals to [0,1]
double Xp, Stt;
    // Arrays to store precomputed values used in the mapping to [0,1).
    double[] meanVG;   
    double[] stdVG; 
    double[] meansumVG;   
    double[] stdsumVG; 
    double[] meandeltaG;   
    double[] stddeltaG; 
    double[] meanGeoVG;           // Mean of log of geometric average up to each step.
    double[] stdevGeoVG;          // Variance of log of geometric average up to each step.
	double teta;
	double nu;
	double muu;
	double St,path, G, X, XX, GG; // SVP: changer le nom de variable path
	double mu2OverNu, muOverNu ;
	double mu2dtOverNu;
	/*
	 * GeometricVarianceGammaProcess GYG; VarianceGammaProcess YG; BrownianMotion X;
	 * GammaProcess G;
	 */
	double[] obsTimes;
	 double deltaG ;
	/* GeometricVarianceGammaProcess GYG;
		VarianceGammaProcess YG;
		BrownianMotion X;
		GammaProcess G;
		*/
    public VGAsianOptionComparable (double r, int d, double t1, double T, double K,double nu,double muu,double teta,
    		 double s0, double sigma, RealsTo01Map map) {
        this.r = r;
        this.d = d;
        
        this.td = T;
        this.strike = K;
        this.s0 = s0;
        this.sigma = sigma;
        this.nu=nu;
        this.teta=teta;
        this.muu=muu;
        this.map = map;
        S0 = s0;
        
        
        discount = Math.exp(-r * td);
        
        obsTimes = new double[d+1];
        obsTimes[0] = 0.0;
        obsTimes[1] = t1;
        double deltaT = (T - t1) / (double)(d - 1);
        for (int j=2; j<=d; j++)
        	 obsTimes[j] = t1 + (j - 1) * deltaT;
        	//obsTimes[j] = (240-d+j)/365;
        	//obsTimes[j] = (230+j)/365;
     /*   G=new GammaProcess(0,muu,nu,(RandomStream)null); // eviter la recreation des memes objets exactements; c'est couteux		
		X=new BrownianMotion(0,teta,sigma,(RandomStream)null);
		YG=new VarianceGammaProcess(0,X,G);
		GYG=new GeometricVarianceGammaProcess(s0,r,YG);		
       */
		stateDim = 3;

        //St = 0;
        meanVG = new double[d+1];
        stdVG = new double[d+1];
        meansumVG = new double[d+1];
        stdsumVG = new double[d+1];
        meanGeoVG = new double[d+1];
        stdevGeoVG = new double[d+1];
        meandeltaG = new double[d+1];
        stddeltaG = new double[d+1];
        w = Math.log(1 - (teta*nu) - Math.pow(sigma, 2)*nu*0.5)/nu;
        //double sumSqj = 0.0;
        meanVG[0]=0.0;
        stdVG[0]=0.0;
        meanGeoVG[0]=0.0;
        stdevGeoVG[0]=0.0;
        meansumVG[0]=0.0;
        stdsumVG[0]=0.0;
        meandeltaG [0] = 0.0 ;
        stddeltaG [0] = 0.0 ;
       /* for (int j=1; j<=d; j++) {
        	meanVG[j]  = teta*obsTimes[j];
        	stdVG[j]=Math.sqrt((teta*teta*nu+sigma*sigma)*obsTimes[j]);
        	meanGeoVG[j] = S0 * Math.exp((r+w+teta+((teta*teta*nu)+sigma*sigma)/2)*obsTimes[j]);
        	//sumSqj += (j-1)^2;
        	stdevGeoVG[j] = S0 * Math.sqrt ((Math.exp(((teta*teta*nu)+sigma*sigma))*obsTimes[j])-1)*Math.exp((2*(r+w+teta)+((teta*teta*nu)+sigma*sigma))*obsTimes[j]);
        	for (int s = 1; s <= j; ++s) {
        	meansumVG[j] += meanGeoVG[s];
        	//stdsumVG[j]+=Math.pow(stdevGeoVG[s],2)/(j*j);
        	stdsumVG[j]+=stdevGeoVG[s]/(double)(j*j);
        	 }
        	stdsumVG[j]=Math.sqrt(stdsumVG[j]);
        }
        */
        
    	
        for (int j=1; j<=d; j++) {        	
        	meanGeoVG[j] = Math.log(S0) + ((w+r) +muu * muu )* obsTimes[j]  ;
        	meansumVG[j] += meanGeoVG[j];
        	meandeltaG[j] = (obsTimes[j] -obsTimes[j-1]) *muu;
        	stddeltaG[j] += Math.sqrt((obsTimes[j] -obsTimes[j-1]) *nu);
        	double gam1 = gamma (obsTimes[j] * muu*muu/(double)nu +0.5);
        	double gam2 = gamma (obsTimes[j] * muu*muu/(double)nu );
        	double gam = gam1 * gam1/(double)(gam2 * gam2);
        	double tt = nu/(double)(obsTimes[j] *muu*muu);
        	double rr =muu * muu* muu /(double) (nu * nu) ;
        	stdevGeoVG[j]= Math.sqrt (nu * muu * obsTimes[j] + sigma * obsTimes[j]*rr * (1-tt ));
        	
        	stdsumVG[j] +=  ((nu * muu * obsTimes[j] + sigma * obsTimes[j]*rr * (1-tt )));
        	stdsumVG[j] = Math.sqrt (stdsumVG[j]);
        }
    }

/*public void generatePathBGSS(RandomStream stream,int step) {
		
		G=new GammaProcess(0,muu,nu,stream);
		
		X=new BrownianMotion(0,teta,sigma,stream);
	
		YG=new VarianceGammaProcess(0,X,G);
		YG.setObservationTimes(obsTimes[step], 1);
		GYG=new GeometricVarianceGammaProcess(s0,r,YG);		
		GYG.setObservationTimes(obsTimes[step], 1);
		GYG.setStream(stream);
		path=YG.generatePath();
		
		St=GYG.generatePath();	
		//double time =	(double)(step)/(double)d;
		//St = s0*Math.exp(r*time + path[0] + w*time);
		
	}*/

/*public void generatePathBGBS(RandomStream stream, int step) {
	
	G=new GammaProcessSymmetricalBridge(0,muu,nu,stream);

	X=new BrownianMotionBridge(0,teta,sigma,stream);
	
	YG=new VarianceGammaProcess(0,X,G);
	
	GYG=new GeometricVarianceGammaProcess(s0,r,YG);	
	
	GYG.setObservationTimes(obsTimes[step], 1);
	
	path=GYG.generatePath();	
	
	double time =	(double)(step)/(double)d;	
	St = s0*Math.exp(r*time + path[0] + w*time);
	
}



	public void generatePathDGBS(RandomStream stream,int step) {
		
		G=new GammaProcessSymmetricalBridge(0,muu,nu,stream);
	
		X=new BrownianMotionBridge(0,teta,sigma,stream);
	
		
		YG=new VarianceGammaProcessDiff(0,teta,sigma,nu,stream);
		
		GYG=new GeometricVarianceGammaProcess(s0,r,YG);
		
		GYG.setObservationTimes(obsTimes[step], 1);		
		path=GYG.generatePath();	
		
		    double time =	(double)(step)/(double)d;
			St = s0*Math.exp(r*time + path[0] + w*time);

	}*/
    
   public double calcFractionPositivePayoff (int n) {
        TallyStore statRuns = new TallyStore();
        simulRunsWithSubstreams(n, d, new MRG32k3a(), statRuns);
        statRuns.quickSort();
        return (n - statRuns.getDoubleArrayList().lastIndexOf(0.0) - 1.0) / n;
    }

    // Initial value of underlying Brownian motion is zero.
    public void initialState() {
        step = 0;
        St = s0;
       G = 0.0;;
        X = 0.0;
        sumS = 0.0;
        path = 0.0;
        logsumS = 0.0;
        substep = 0;
    }

    // Simulates the next step.
//    public void nextStep(RandomStream stream) {
//    	
//    	 
//    	 if (substep == 0) {
//    	 
//    		 //System.out.println("ok1");
//    	 muOverNu  = muu / (double)nu;
//    	 mu2OverNu = muu * muu / (double)nu;
//    	 mu2dtOverNu = mu2OverNu * (obsTimes[step+1]- obsTimes[step]);
//    		
//    	 //deltaG = GammaDist.inverseF(mu2dtOverNu, muOverNu, 1, stream.nextDouble());
//    	deltaG = GammaGen.nextDouble(stream, mu2dtOverNu, muOverNu);
//    	
//    	 substep++;
//    	 }
//    	 else  if (substep == 1)  {
//    		 //System.out.println("ok2");
//    	 GG= G + deltaG;
//    	 G =GG;
//    	 Xp =X;
//         XX = X + teta* deltaG + sigma *Math.sqrt(deltaG)*NormalDist.inverseF01(stream.nextDouble());
//         X =XX;
//         Stt = St*Math.exp((r+w)*(obsTimes[step+1]- obsTimes[step])+ X-Xp);
//         St =Stt;
//         sumS += St;
//         logsumS  += Math.log (St);
//         substep=0;
//         step++; 
//       //  System.out.println("step \t" + step);
//    	 }
//    
//   
//    }

    public void nextStep(RandomStream stream) {
    	
   	 
    	step++;
   	 
   		 //System.out.println("ok1");
   	 muOverNu  = muu / (double)nu;
   	 mu2OverNu = muu * muu / (double)nu;
   	 mu2dtOverNu = mu2OverNu * (obsTimes[step]- obsTimes[step-1]);
   		
   	 //deltaG = GammaDist.inverseF(mu2dtOverNu, muOverNu, 1, stream.nextDouble());
   	deltaG = GammaGen.nextDouble(stream, mu2dtOverNu, muOverNu);
   	
   
 
   		 //System.out.println("ok2");
   	 GG= G + deltaG;
   	 G =GG;
   	 Xp =X;
        XX = X + teta* deltaG + sigma *Math.sqrt(deltaG)*NormalDist.inverseF01(stream.nextDouble());
        X =XX;
        Stt = St*Math.exp((r+w)*(obsTimes[step]- obsTimes[step-1])+ X-Xp);
        St =Stt;
        sumS += St;
        logsumS  += Math.log (St);

      //  System.out.println("step \t" + step);
   	 
   
  
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
        if (!(other instanceof VGAsianOptionComparable)) {
            throw new IllegalArgumentException("Can't compare a "
                    + "VG with other types of Markov chains.");
        }
        double x, mx;
        switch (j) {
            case 0:
                mx = ((VGAsianOptionComparable) other).St;
                return (St > mx ? 1 : (St < mx ? -1 : 0));
            case 1:
             /*   x = sumS;
                mx = ((VGAsianOptionComparable) other).sumS;
                return (x > mx ? 1 : (x < mx ? -1 : 0));*/
                x = sumS / (double) step;
    			mx = ((VGAsianOptionComparable) other).sumS / (double) ((VGAsianOptionComparable) other).step;
    			return (x > mx ? 1 : (x < mx ? -1 : 0));

          case 2:
                mx = ((VGAsianOptionComparable) other).deltaG;
               return (deltaG > mx ? 1 : (deltaG < mx ? -1 : 0));
            default:
                throw new IllegalArgumentException("Invalid state index");
        }
    }


    // Returns the state transformed to (0,1)^2.
    public double[] getPoint() {
    	double[] state01 = new double[3];
        state01[0] = getCoordinate(0);
        state01[1] = getCoordinate(1);
        state01[2] = getCoordinate(2);
        return state01;
    }

    // Returns coordinate j of the state mapped to (0,1).
  /*  public double getCoordinate (int j) {
        double zvalue;
        switch (j) {
            case 0:
                zvalue = St-meanGeoVG[step];
                zvalue /= stdevGeoVG[step];
                return map.realTo01 (zvalue);
             
            case 1:
                zvalue = sumS/step - meansumVG[step];
                zvalue /= stdsumVG[step];
                return map.realTo01 (zvalue);
              
//            case 2:
//            	zvalue = path-meanVG[step];
//                zvalue /= stdVG[step];
//                return map.realTo01 (zvalue);
            default:
                throw new IllegalArgumentException("Invalid state index");
        }
    }*/
    
    public double getCoordinate (int j) { //TODO: step-1???
        double zvalue;
        switch (j) {
            case 0:
                zvalue = Math.log(St)-meanGeoVG[step];
                zvalue /= stdevGeoVG[step];
                return map.realTo01 (zvalue);
             
            case 1:
                zvalue = Math.log(sumS/(double)step) - meansumVG[step];
                zvalue /= stdsumVG[step];
                return map.realTo01 (zvalue);
            case 2:
                zvalue = deltaG - meandeltaG[step];
                zvalue /= stddeltaG[step];
                return map.realTo01 (zvalue);
              
            default:
                throw new IllegalArgumentException("Invalid state index");
        }
    }
    public double[] getState() {
		double [] state = {St,sumS/(double)step};

//		double [] state = {St,sumS/(double)step, deltaG};
		return state;
	}

	@Override
	public int dimension() {
		// TODO Auto-generated method stub
		return 2;
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
