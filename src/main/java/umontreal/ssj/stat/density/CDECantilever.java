package umontreal.ssj.stat.density;

import java.util.Arrays;

import umontreal.ssj.probdist.NormalDist;

public class CDECantilever extends ConditionalDensityEstimator {
	
	/**
	 * Weights for the variables E, X, and Y (in this order).
	 */
	private double[] weights;
	
	private double alpha, beta, gamma; //alpha = 4L^3/(wt); beta = w^4; gamma = t^4;
	
	private double  muE, sigmaE,  muX,  sigmaX,  muY,  sigmaY;
	
	/**
	 * Constructor
	 * @param L
	 * @param t
	 * @param w
	 * @param muE
	 * @param sigmaE
	 * @param muX
	 * @param sigmaX
	 * @param muY
	 * @param sigmaY
	 */
	public CDECantilever(double L, double t, double w, double muE, double sigmaE, double muX, double sigmaX, double muY, double sigmaY) {
		this.weights = new double[3];
		Arrays.fill(weights, 1.0/3.0);
		this.alpha = 4.0 * L * L * L/ (w * t );
		this.beta = w * w * w* w;
		this.gamma = t * t * t * t;
		this.muE = muE;
		this.sigmaE = sigmaE;
		this.muX = muX;  
		this.sigmaX = sigmaX; 
		this.muY = muY;  
		this.sigmaY = sigmaY;
	}
	
	
	public CDECantilever(double L, double t, double w,double muE, double sigmaE, double muX, double sigmaX, double muY, double sigmaY,double[][] data, double[] weights) {
		this(L, t, w, muE, sigmaE,  muX,  sigmaX,  muY,  sigmaY);
		setData(data);
		setWeights(weights);
	}
	
	public CDECantilever(double L, double t, double w,double muE, double sigmaE, double muX, double sigmaX, double muY, double sigmaY,double[] weights) {
		this(L, t, w, muE, sigmaE,  muX,  sigmaX,  muY,  sigmaY);
		setWeights(weights);
	}
	
	/**
	 * Setter for the weights. Normalizes the weights to 1 in \f$\ell^1\f$.
	 * @param weights
	 */
	public void setWeights(double[] weights) {
		this.weights = new double[3];
		for(int i = 0; i < 3; i++)
			this.weights[i] = weights[i];
//		normalizeWeights();
	}
	
	private void normalizeWeights() {
		double sum = 0.0;
		for(double w:weights)
			sum+=w;
		for(int i = 0; i < 3; i++)
			weights[i] /= sum;
	}
	
	private double deltaSqX(double x, double[] data) {
		return ( beta * (x * x * data[0] * data[0] /(alpha * alpha) - data[2]*data[2]/gamma)  );
	}
	
	private double deltaSqY(double x, double[] data) {
		return ( gamma * (x * x * data[0] * data[0] /(alpha * alpha) - data[1]*data[1]/beta)  );
	}

	@Override
	public double evalEstimator(double x, double[] data) {
		double val = 0.0;
		double arg;
		double temp;

		if(weights[0]>=0) {
		arg = alpha * Math.sqrt(data[1] * data[1] / beta + data[2] * data[2] / gamma ) / x;
		val += weights[0] * NormalDist.density(muE, sigmaE, arg) * arg /x;
		}
//		System.out.println("TEST:\t" +arg + "\t" + val + "\t" + muE + "\t" + sigmaE + "\t" + x);
		
		temp = deltaSqX(x,data);
		
		if( (weights[1] >= 0) && (temp>=0) ) {
		arg = Math.sqrt(temp );
		temp = weights[1] * x * data[0] * data[0] * beta  /(alpha * alpha * arg);
		val += temp * ( NormalDist.density(muX, sigmaX, arg) + NormalDist.density(muX, sigmaX, -arg) );
		}
//		else {
//			System.out.println("Expression for f_{D|E,Y} not admissible");
//		}
//		System.out.println("TEST:\t" +arg + "\t" + val + "\t" + muX + "\t" + sigmaX + "\t" + x);

		temp = deltaSqY(x,data);
		if(weights[2] >= 0) {
			arg = Math.sqrt(temp );
			temp = weights[2] * x * data[0] * data[0] * gamma  /(alpha * alpha * arg);
			val += temp * ( NormalDist.density(muY, sigmaY, arg) + NormalDist.density(muY, sigmaY, -arg) );
			}
		
		return val;
	}
	
	@Override
	public String toString() {
		return "CDECantilever";
	}


}
