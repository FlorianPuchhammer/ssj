package umontreal.ssj.mcqmctools.examples;

import java.util.List;

import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.rng.RandomStream;

public class creditMetrics implements MonteCarloModelDoubleArray {

	/**
	 * forward interest rates \f$f_{0,k}\f$ (in %). The first index represents the
	 * rating (AAA, AA, ...,CCC) and the second index the number of years (1 year to
	 * 9 years).
	 */
	static final double[][] fwdInterest0 = { { 0.045, 0.014, 0.112, 0.25, 0.452, 0.637, 0.736, 1.052, 1.302 },
			{ 0.219, 0.324, 0.438, 0.517, 0.686, 0.93, 1.048, 1.196, 1.34 },
			{ 0.4, 0.536, 0.61, 0.795, 0.946, 1.112, 1.266, 1.463, 1.617 },
			{ 0.7, 0.886, 1.227, 1.213, 1.325, 1.651, 1.599, 2.091, 2.034 },
			{ 1, 1.05, 1.1, 1.25, 1.45, 1.7, 1.65, 1.8, 2.05 }, { 1.2, 1.1, 1.15, 1.35, 1.65, 1.75, 1.8, 1.85, 2.2 },
			{ 1.45, 1.141, 1.072, 1.569, 1.994, 1.849, 1.898, 1.756, 2.231 } };

	public static void printFwdInterest1fromFormula() {
		StringBuffer sb = new StringBuffer("{");
		double[][] interest = new double[7][8];
		for(int j = 0; j < 7; j++) {
			sb.append("{");
			for(int i = 0; i <8; i++) {
				
				interest[j][i] = fwdInterest(0,i+1,j)*100.0;
				sb.append(interest[j][i] + ",");
			}
			sb.deleteCharAt(sb.length()-1);
			sb.append("},\n");
		}
		sb.deleteCharAt(sb.length()-1);
		sb.append("}");
		System.out.println(sb.toString());
	}
	
	static final double[][] fwdInterest1 = {
		{-0.016990394322546898,0.14551682430061774,0.3184266377468159,0.55400844749105,0.7558197060731686,0.8516297880702295,1.1966821971329145,1.4602314265983685},
		{0.42911000908010255,0.5476793954245052,0.6165301143675306,0.8030896241202345,1.072804158095142,1.186831831480295,1.3363467773249482,1.4810037835732537},
		{0.672184223107597,0.7151646587594929,0.9270117060945049,1.08296332117519,1.2550047669693143,1.4110578191913747,1.6157731258617858,1.770158595320881},
		{1.0723435551142302,1.4915333437111489,1.3845800986505896,1.4818551708841499,1.842275034372376,1.7496116951508656,2.2912766497985793,2.2019877810747657},
		{1.1000247524752371,1.1500371225882766,1.3334707715224825,1.56281292525009,1.8405811055876686,1.758739305346313,1.9148018241495368,2.18201520412733},
		{1.0000988142292222,1.1250092645969545,1.4000493908509215,1.7628123075063007,1.860358177070931,1.9003452817635935,1.9431973275509051,2.325692794337697},
		{0.8329411631345263,0.8835284851695935,1.6086976776967798,2.1304551814432138,1.9289881135448494,1.972858771167152,1.799789565921306,2.329046803155199}
		};
	
	/**
	 * The first index represents the security class (Senior secured, senior
	 * unsecured, senior subordinated, subordinated, junior subordinated) and the
	 * second index represents the recovery rate in % for the mean (index 0) and of
	 * the standard deviation (index 1).
	 */
	static final double[][] securityClassesRecoveryRates = { { 53.8, 26.86 }, { 51.13, 25.45 }, { 38.52, 23.81 },
			{ 32.74, 20.18 }, { 17.09, 10.9 } };

	/**
	 * The probabilities \f$p_{Y\to X}\f$, i.e., the probability that a credit with
	 * current rating \f$Y\f$ will be rated \f$X\f$ in one year. The first index
	 * gives the rating \f$\Yf$ and takes the values (AAA, AA,...,CCC), and the
	 * second index represents the rating \f$X\f$ with the values (AAA,AA,...,D).
	 */
	static final double[][] transitionProbabilities = { { 90.81, 8.33, 0.68, 0.06, 0.12, 0, 0, 0 },
			{ 0.7, 90.65, 7.79, 0.64, 0.06, 0.14, 0.02, 0 }, { 0.09, 2.27, 91.05, 5.52, 0.74, 0.26, 0.01, 0.06 },
			{ 0.02, 0.33, 5.95, 86.93, 5.3, 1.17, 0.12, 0.18 }, { 0.03, 0.14, 0.67, 7.73, 80.53, 8.84, 1, 1.06 },
			{ 0, 0.11, 0.24, 0.43, 6.48, 83.46, 4.07, 5.2 }, { 0.22, 0, 0.22, 1.3, 2.38, 11.24, 64.86, 19.79 } };

	/**
	 * Integration limits \f$Z_X^Y\f$, where the first index corresponds to rating \f$Y\f$ and the second one to
	 * rating \f$X\f$.
	 */
	static final double[][] limitsZ = {
			{ Double.POSITIVE_INFINITY, -1.3291454135482899, -2.3824043423823116, -2.9112377262430056,
					-3.0356723666270735, -4.0, -5.0, -6.0, Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 2.4572633902054375, -1.3626273014608778, -2.3824043423823116,
					-2.847963287487918, -2.9478425521849063, -3.540083799206145, -5.0, Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 3.1213891493598656, 1.9845011501354217, -1.5070415784970754, -2.300851965340215,
					-2.7163805834608654, -3.1946510537632866, -3.2388801183529776, Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 3.540083799206145, 2.6968442608781262, 1.5300675881378285, -1.4931420783259008,
					-2.1780810922893408, -2.747781385444993, -2.9112377262430056, Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 3.431614403623269, 2.929049748937626, 2.391055785778313, 1.367719160636112,
					-1.231863708734983, -2.0415116207180075, -2.3044035663594626, Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 3.7190164854556804, 3.0356723666270735, 2.6874494471514727, 2.4135029495884788,
					1.455973302594268, -1.3243103335876891, -1.6257633862332346, Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 2.862736263505904, 2.862736263505904, 2.6275587101050375, 2.113008971608029,
					1.738061374074731, 1.0215371869194867, -0.8491461019870323, Double.NEGATIVE_INFINITY } };

	/**
	 * Forward interest rates \f$f_{k,k+l}\f$ for credits with rating \a rating.
	 * @param k current year.
	 * @param l number of years in the future.
	 * @param rating the current rating.
	 * @return the interest rate from year \f$k\f$ to \f$(k+l)\f$ considering rating \a rating.
	 */
	public static double fwdInterest(int k, int l, int rating) {
		double invL = 1.0/(double)l;
		double num = Math.pow(1.0 + fwdInterest0[rating][k+l] * 0.01 , (double) (k+l+1)*invL);
		double denom =Math.pow( 1.0 + fwdInterest0[rating][k] * 0.01 , (double) (k+1)*invL);
		return num/denom -1.0;
	}
	
	public static double A0(Credit K) {
		double a0 = K.getAmount()/(Math.pow(1.0 + fwdInterest0[K.getRating()][K.getDuration()-1]*0.01, (double)K.getDuration()));
		for(int i = 1; i <= K.getDuration(); i++) {
			a0 += K.getCoupon()*0.01 * K.getAmount()/ Math.pow(1.0 + fwdInterest0[K.getRating()][i-1], (double) i);
		}
		return a0;
	}
	
	public static double GesA0(List<Credit> listK) {
		double gesA0 = 0.0;
		for(Credit K : listK)
			gesA0 += A0(K);
		return gesA0;
	}
	
	public double A1(Credit K, int rating) {
		double a1 = 0.0;
		if(rating != 7) {
			a1 = K.getAmount()/Math.pow(1.0 + fwdInterest1[rating][K.getDuration() -1]*0.01, (double)K.getDuration() - 1.0);
			for(int i = 1; i < K.getDuration(); i++)
				a1 += K.getCoupon() * 0.01 * K.getAmount() / Math.pow(1.0 + fwdInterest1[rating][i-1]*0.01, (double) i);
		}
		else 
			a1 = 0.01 * securityClassesRecoveryRates[K.getSecurityClass()][0] * K.getAmount();
		
		return a1;
	}
	@Override
	public void simulate(RandomStream stream) {
		// TODO Auto-generated method stub

	}

	@Override
	public double[] getPerformance() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getPerformanceDim() {
		// TODO Auto-generated method stub
		return 0;
	}
	
	public static void main(String[] args) {
		StringBuffer sb = new StringBuffer("{");
		double[][] zinsen = new double[7][8];
		for(int j = 0; j < 7; j++) {
			sb.append("{");
			for(int i = 0; i <8; i++) {
				
				zinsen[j][i] = fwdInterest(0,i+1,j)*100.0;
				sb.append(zinsen[j][i] + ",");
			}
			sb.deleteCharAt(sb.length()-1);
			sb.append("},\n");
		}
		sb.deleteCharAt(sb.length()-1);
		sb.append("}");
		System.out.println(sb.toString());
	}
	
	private double getBetaVariate(Credit K, double p, double q, int maxIts,RandomStream stream) {
		double pTemp, qTemp;
		for(int i = 0; i < maxIts; i++) {
			
		}
	}

}
