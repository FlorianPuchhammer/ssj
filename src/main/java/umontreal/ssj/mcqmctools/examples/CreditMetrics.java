package umontreal.ssj.mcqmctools.examples;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.probdist.BetaDistFlo;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.randvarmulti.MultinormalPCAGen;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;

public class CreditMetrics implements MonteCarloModelDouble{

	/**
	 * Forward interest rates \f$f_{0,k}\f$ (in %). The first index represents the
	 * rating (AAA, AA, ...,CCC) and the second index the number of years (1 year to
	 * 9 years). Note: the index corresponds to the number of years minus 1.
	 */
	static final double[][] fwdInterest0 = { { 0.045, 0.014, 0.112, 0.25, 0.452, 0.637, 0.736, 1.052, 1.302 },
			{ 0.219, 0.324, 0.438, 0.517, 0.686, 0.93, 1.048, 1.196, 1.34 },
			{ 0.4, 0.536, 0.61, 0.795, 0.946, 1.112, 1.266, 1.463, 1.617 },
			{ 0.7, 0.886, 1.227, 1.213, 1.325, 1.651, 1.599, 2.091, 2.034 },
			{ 1, 1.05, 1.1, 1.25, 1.45, 1.7, 1.65, 1.8, 2.05 }, { 1.2, 1.1, 1.15, 1.35, 1.65, 1.75, 1.8, 1.85, 2.2 },
			{ 1.45, 1.141, 1.072, 1.569, 1.994, 1.849, 1.898, 1.756, 2.231 } };

	/**
	 * Forward interest rates \f$f_{1,1+k}\f$ (in %). The first index represents the
	 * current rating and the second index the number of years \f$k\f$ (1 year to 9
	 * years). Note: the index corresponds to the number of years in the future
	 * minus 1.
	 */
	static final double[][] fwdInterest1 = {
			{ -0.016990394322546898, 0.14551682430059554, 0.3184266377467937, 0.5540084474910278, 0.7558197060731464,
					0.8516297880702295, 1.1966821971329145, 1.4602314265983685 },
			{ 0.42911000908010255, 0.547679395424483, 0.6165301143675528, 0.8030896241202345, 1.072804158095142,
					1.186831831480295, 1.3363467773249704, 1.4810037835732537 },
			{ 0.672184223107597, 0.7151646587594929, 0.9270117060945049, 1.082963321175212, 1.2550047669693365,
					1.4110578191913747, 1.6157731258617858, 1.770158595320881 },
			{ 1.0723435551142302, 1.4915333437111489, 1.3845800986505674, 1.4818551708841277, 1.842275034372376,
					1.7496116951508656, 2.2912766497985793, 2.2019877810747657 },
			{ 1.1000247524752371, 1.1500371225882766, 1.3334707715224825, 1.56281292525009, 1.8405811055876464,
					1.758739305346313, 1.9148018241495368, 2.18201520412733 },
			{ 1.0000988142292222, 1.1250092645969767, 1.4000493908509215, 1.762812307506323, 1.8603581770709088,
					1.9003452817636157, 1.9431973275509273, 2.325692794337675 },
			{ 0.8329411631345485, 0.8835284851695713, 1.608697677696802, 2.130455181443236, 1.9289881135448494,
					1.972858771167152, 1.799789565921306, 2.329046803155199 } };

	/**
	 * The first index represents the security class (Senior secured, senior
	 * unsecured, senior subordinated, subordinated, junior subordinated) and the
	 * second index represents the recovery rate in % for the mean (index 0) and of
	 * the standard deviation (index 1).
	 */
	static final double[][] recoveryRates = { { 53.8, 26.86 }, { 51.13, 25.45 }, { 38.52, 23.81 }, { 32.74, 20.18 },
			{ 17.09, 10.9 } };

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
	 * Integration limits \f$Z_X^Y\f$, where the first index corresponds to rating
	 * \f$Y\f$ and the second one to rating \f$X\f$.
	 */
	static final double[][] limitsZ = {
			{ Double.POSITIVE_INFINITY, -1.32915, -2.3824, -2.91124, -3.03567, -4, -5, -6, Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 2.45726, -1.36263, -2.3824, -2.84796, -2.94784, -3.54008, -5,
					Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 3.12139, 1.9845, -1.50704, -2.30085, -2.71638, -3.19465, -3.23888,
					Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 3.54008, 2.69684, 1.53007, -1.49314, -2.17808, -2.74778, -2.91124,
					Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 3.43161, 2.92905, 2.39106, 1.36772, -1.23186, -2.04151, -2.3044,
					Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 3.71902, 3.03567, 2.68745, 2.4135, 1.45597, -1.32431, -1.62576,
					Double.NEGATIVE_INFINITY },
			{ Double.POSITIVE_INFINITY, 2.86274, 2.86274, 2.62756, 2.11301, 1.73806, 1.02154, -0.849146,
					Double.NEGATIVE_INFINITY } };

	/**
	 * The credit portfolio.
	 */
	public List<Credit> portfolio;
	/**
	 * The covariance matrix for the portfolio
	 */
	public DoubleMatrix2D trafoMat;
	/**
	 * Number of credits in the portfolio.
	 */
	public int dimension;
	/**
	 * future ratings
	 */
	public int[] ratings;

	/**
	 * random stream for generating evaluation points, beta-variates, etc.
	 */
	private RandomStream utilStream;

	/**
	 * Constructor that reads the individual credits from the file \a filename.
	 * 
	 * @param filename
	 *            the file containing the credits of the portfolio.
	 * @param sigma
	 *            the covariance matrix.
	 * @param utilStream
	 *            the stream to be used as utility-stream
	 * @throws FileNotFoundException
	 */
	public CreditMetrics(String filename, double[][] sigma, RandomStream utilStream) throws FileNotFoundException {
		readPortfolio(filename);
		this.trafoMat = MultinormalPCAGen.decompPCA(sigma);
		this.dimension = portfolio.size();
		this.utilStream = utilStream;
	}

	/**
	 * Same as #CreditMetrics(String, double[][], RandomStream), but here the
	 * credits are uncorrelated.
	 * 
	 * @param filename
	 *            the file containing the credits of the portfolio.
	 * @param utilStream
	 *            the stream to be used as utility-stream
	 * @throws FileNotFoundException
	 */
	public CreditMetrics(String filename, RandomStream utilStream) throws FileNotFoundException {
		readPortfolio(filename);
		this.dimension = portfolio.size();
		this.trafoMat = DoubleFactory2D.sparse.identity(dimension);
		this.utilStream = utilStream;
	}

	/**
	 * Same as #CreditMetrics(String, double[][], RandomStream) but here, the
	 * utility stream is created as a new instance of a #MRG32k3a.
	 * 
	 * @param filename
	 *            the file containing the credits of the portfolio.
	 * @param sigma
	 *            the covariance matrix.
	 * @throws FileNotFoundException
	 */
	public CreditMetrics(String filename, double[][] sigma) throws FileNotFoundException {
		this(filename, sigma, new MRG32k3a());
	}

	/**
	 * Same as #CreditMetrics(String, double[][], RandomStream) but here, the
	 * utility stream is created as a new instance of a #MRG32k3a and the credits of
	 * the portfolio are uncorrelated.
	 * 
	 * @param filename
	 *            the file containing the credits of the portfolio.
	 * @throws FileNotFoundException
	 */
	public CreditMetrics(String filename) throws FileNotFoundException {
		this(filename, new MRG32k3a());
	}

	/**
	 * Reads credits from a file and stores them in the portfolio.This function is
	 * tailored to take input from the Mathematica-files. To adjust the indices
	 * accordingly, the rating, the security class, and the sector are reduced by 1.
	 * 
	 * @param filename
	 *            the file containing the credits of the portfolio.
	 * @throws FileNotFoundException
	 */
	private void readPortfolio(String filename) throws FileNotFoundException {
		Scanner sc = new Scanner(new BufferedReader(new FileReader(filename)));
		portfolio = new ArrayList<Credit>();
		String[] line;
		while (sc.hasNextLine()) {
			line = sc.nextLine().trim().split("\t");
			portfolio.add(
					new Credit(Double.parseDouble(line[0]), Integer.parseInt(line[1]) - 1, Integer.parseInt(line[2]),
							Double.parseDouble(line[3]), Integer.parseInt(line[4]) - 1, Integer.parseInt(line[5]) - 1));
		}
		sc.close();
	}

	/**
	 * Constructs a portfolio with the credits from \a portfolio, covariance matrix
	 * \a sigma, and the utility stream \a utilStream.
	 * 
	 * @param portfolio
	 *            the list of credits in the portfolio.
	 * @param sigma
	 *            the covariance matrix.
	 * @param utilStream
	 *            the stream used as utility stream.
	 */
	public CreditMetrics(List<Credit> portfolio, double[][] sigma, RandomStream utilStream) {
		this.portfolio = portfolio;
		this.trafoMat = MultinormalPCAGen.decompPCA(sigma);
		this.dimension = portfolio.size();
		this.utilStream = utilStream;
	}

	/**
	 * Same as #CreditMetrics(List, double[][], RandomStream) but with a new
	 * instance of #MRG32k3a as the utility stream.
	 * 
	 * @param portfolio
	 *            the list of credits in the portfolio
	 * @param sigma
	 *            the covariance matrix.
	 */
	public CreditMetrics(List<Credit> portfolio, double[][] sigma) {
		this(portfolio, sigma, new MRG32k3a());
	}

	/**
	 * Computes \f$A_0\f$ for one credit \a cred.
	 * 
	 * @param cred
	 *            the credit.
	 * @return \f$A_0\f$ of that credit.
	 */
	public static double A0(Credit cred) {
		double a0 = cred.getAmount() / (Math.pow(1.0 + fwdInterest0[cred.getRating()][cred.getDuration() - 1] * 0.01,
				(double) cred.getDuration()));
		for (int i = 1; i <= cred.getDuration(); i++) {
			a0 += cred.getCoupon() * 0.01 * cred.getAmount()
					/ Math.pow(1.0 + fwdInterest0[cred.getRating()][i - 1] * 0.01, (double) i);
		}
		return a0;
	}

	/**
	 * Computes the value \f$A_0\f$ for the entire portfolio.
	 * 
	 * @return \f$A_0\f$ for the entire portfolio.
	 */
	public double A0() {
		double sum = 0.0;
		for (Credit C : portfolio) {
			sum += A0(C);
		}
		return sum;
	}

	/**
	 * Computes the value \f$A_1\f$ of the credit \a cred with future rating \a
	 * rating.
	 * 
	 * @param cred
	 *            the credit.
	 * @param rating
	 *            the future rating.
	 * @param stream
	 *            the random stream (to generate a Beta-variate).
	 * @return the value \f$A_1\f$ of \a cred.
	 */
	//TODO: check if no. of iteration changes much
	public static double A1(Credit cred, int rating, RandomStream stream) {

		double a1 = 0.0;
		if (rating != 7) {
			// TODO: indices w.r.t. duration correct???
			a1 = cred.getAmount() / Math.pow(1.0 + fwdInterest1[rating][cred.getDuration() - 2] * 0.01,
					(double) cred.getDuration() - 1.0);
			for (int i = 1; i < cred.getDuration(); i++)
				a1 += cred.getCoupon() * 0.01 * cred.getAmount()
						/ Math.pow(1.0 + fwdInterest1[rating][i - 1] * 0.01, (double) i);
		} else
			a1 = getBetaVariate(cred, 0.5, 0.5, 20, stream);

		return a1;
	}
	
	/**
	 * Based on the future ratings \a rating, this function computes the value \f$A_1\f$ for the entire portfolio.
	 * @param rating the future ratings.
	 * @return \f$A_1\f$ of the portfolio.
	 */
	public double A1(int [] rating) {
		double sum = 0.0;
		for(int j = 0; j < rating.length; j++)
			sum += A1(portfolio.get(j),rating[j], utilStream);
		return sum;
	}

	/**
	 * Generates a beta variate whose parameters are the solutions of a certain two-dimensional system of
	 * non-linear equations (TODO: add equation). These equations are solved by newton-iteration with \a maxIts iterations and initial
	 * values \a p and \a q.
	 * 
	 * @param cred the underlying credit.
	 * @param p initial value for the first coordinate for Newton method.
	 * @param q initial value for the second coordinate Newton method.
	 * @param maxIts the maximal number of iterations of the Newton method
	 * @param stream the random stream to draw from the Beta distribution.
	 * @return a beta variate with certain parameters.
	 */
	//TODO: re-check this method/formulas/...!
	private static double getBetaVariate(Credit cred, double p, double q, int maxIts, RandomStream stream) {
		double pTemp, qTemp, denom;
		double c1 = recoveryRates[cred.getSecurityClass()][0] * 0.01;
		double c2 = recoveryRates[cred.getSecurityClass()][1] * recoveryRates[cred.getSecurityClass()][1] * 0.0001;
		for (int i = 0; i < maxIts; i++) {
			denom = 2.0 * c2 * (p + q) * (p + q) * (p + q);
			pTemp = (c2 * ((c1 - 1.0) * p * p + 2.0 * c1 * p * q + c1 * q * q)
					- (p + q) * (p + q) * ((c1 - 1.0) * p * p * (p + 1.0) - p * p * q - c1 * (3.0 * p + 1.0) * q * q
							- 2.0 * c1 * q * q * q))
					/ (q * denom);
			qTemp = (c2 * (-(c1 - 1.0) * p * p - 2.0 * (c1 - 1.0) * p * q - c1 * q * q)
					- (p + q) * (p + q) * (2.0 * (c1 - 1.0) * p * p * p - p * q * q - c1 * q * q * (q + 1.0)
							+ (c1 - 1.0) * p * p * (3.0 * q + 1.0)))
					/ (p * denom);
			p = pTemp;
			q = qTemp;
		}
		return BetaDistFlo.inverseF(p, q, stream.nextDouble()) * cred.getAmount();

	}
	
	
	public static double trueA1(Credit cred, int rating) {
		double a1 = 0.0;
		if (rating != 7) {
			a1 = cred.getAmount()
					/ Math.pow(1.0 + fwdInterest1[rating][cred.getDuration() - 2] * 0.01, (double) cred.getDuration() - 1.0);
			for (int i = 1; i < cred.getDuration(); i++)
				a1 += cred.getCoupon() * 0.01 * cred.getAmount()
						/ Math.pow(1.0 + fwdInterest1[rating][i - 1] * 0.01, (double) i);
		} else
			a1 = 0.01 * recoveryRates[cred.getSecurityClass()][0] * cred.getAmount();

		return a1;
	}
	
	/**
	 * Computes the nominal value of the portfolio.
	 * 
	 * @return the nominal value.
	 */
	public double nom() {
		double sum = 0.0;
		for (Credit cred : portfolio) {
			sum += cred.getAmount();
		}
		return sum;
	}
	
	/**
	 * Normalizes the credit portfolio using the normalizing value \a nomVal.
	 * @param nomVal the value w.r.t. which to normalize.
	 */
	public void normalize(double nomVal) {
		double val;
		for (int j = 0; j < dimension; j++) {
			val = portfolio.get(j).getAmount() * 100.0 / nomVal;
			portfolio.get(j).setAmount(val);
		}
	}
	
	/**
	 * Computes \f$\mathbb{E}A_1\f$ for the credit \a cred.
	 * 
	 * @param cred the credit
	 * @return the expected future value of  \a cred.
	 */
	public static double expectedA1(Credit cred) {
		double sum = transitionProbabilities[cred.getRating()][7] * recoveryRates[cred.getSecurityClass()][0]
				* 0.0001 * cred.getAmount();
		for (int i = 0; i < 7; i++) {
			sum += transitionProbabilities[cred.getRating()][i] * 0.01 * trueA1(cred, i);
		}
		return sum;
	}

	/**
	 * Computes \f$\mathbb{E}A_1\f$ for the portfolio.
	 * @return the expected future value of the portfolio.
	 */
	public double expectedA1() {
		double sum = 0.0;
		for(Credit cred : portfolio)
			sum += expectedA1(cred);
		
		return sum;
	}
	/**
	 * Computes the variance of \f$A1\f$ based on the credit \a cred.
	 * @param cred the underlying credit.
	 * @return the variance of \f$A1\f$ for \a cred.
	 */
	public static double varA1(Credit cred) {
		double ew = expectedA1(cred);
		double diff;
		double fac1 = transitionProbabilities[cred.getRating()][7] * 0.01;
		double fac2 = recoveryRates[cred.getSecurityClass()][0] * cred.getAmount() * 0.01 - ew;
		double fac3 = recoveryRates[cred.getSecurityClass()][1] * cred.getAmount() * 0.01;
		double sum = fac1 * (fac2 * fac2 +  fac3 * fac3);
		for (int i = 0; i < 7; i++) {
			diff = trueA1(cred, i) - ew;
			sum += transitionProbabilities[cred.getRating()][i] * 0.01 * diff * diff;
		}
		return sum;
	}
	
	/**
	 * Computes the variance of \f$A_1\f$ of the entire portfolio.
	 * @return the variance of \f$A_1\f$ of the portfolio.
	 */
	public double varA1() {
		double sum = 0.0;
		for(Credit cred : portfolio) {
			sum += varA1(cred);
		}
		return sum;
	}
	
	/**
	 * Computes the future ratings based on the vector \a y.
	 * @param y
	 */
	public void newRating(double[] y) {
		ratings = new int[dimension];
		Arrays.fill(ratings, -1);
		for (int j = 0; j < dimension; j++) {
			for (int r = 0; r < 8; r++) {
				if (y[j] < limitsZ[portfolio.get(j).getRating()][r])
					ratings[j] = r;
			}
		}
	}
	
	/**
	 * Creates an instance of the future ratings of the portfolio.
	 */
	public void simulate(RandomStream stream) {

		double[] z = new double[dimension];
		nextPoint(z, stream);
		

		newRating(z);
	}
	
	/**
	 * Generates a vector of uniforms, transforms them into standard normals and uses the transformation matrix #trafoMat
	 * to draw from the desired multinormal distribution. The variates are stored in \a z.
	 * @param z the array in which the variates are stored.
	 * @param stream the random stream used.
	 */
	private void nextPoint(double[] z, RandomStream stream) {
		double[] u = new double[dimension];
		for (int i = 0; i < dimension; i++) {
			u[i] = NormalDist.inverseF01(stream.nextDouble());
		}
		for (int j = 0; j < dimension; j++) {
			z[j] = 0;
			for (int c = 0; c < dimension; c++)
				z[j] += trafoMat.getQuick(j, c) * u[c];
		}
	}
	
	/**
	 * Gives the dimension, i.e. the number of credits in the portfolio.
	 * @return the dimension of the problem.
	 */
	public int getDimension() {
		return dimension;
	}
	
	/**
	 * Computes the expected future value of \f$A_1\f$ with future ratings #ratings for this portfolio.
	 */
	public double getPerformance() {
		return A1(ratings);
	}
	
	public String toString() {
		return "CreditMetrics (" + dimension + " Credits)";
	}
	
	public String toStringHeader() {
		StringBuffer sb = new StringBuffer("");
		sb.append("*****************************************************************\n");
		sb.append("* CREDIT METRICS \n");
		sb.append("*\t Number of Credits:\t" + dimension + "\n");
		sb.append("*\t A0:\t\t\t" + A0()+"\n");
		sb.append("*\t Expected A1:\t\t" + expectedA1() + "\n");
		sb.append("*\t Variance A1:\t\t" + varA1()+"\n");
		sb.append("*****************************************************************\n\n");
		return sb.toString();
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		String filename = "data/creditMetrics/KP5/KP5.dat";
		CreditMetrics portfolio = new CreditMetrics(filename);
		portfolio.normalize(portfolio.nom());
		System.out.println(portfolio.toStringHeader());


		
	}

	
}
