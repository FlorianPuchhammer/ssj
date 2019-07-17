package umontreal.ssj.stat.density;

public class LLCantilever extends ConditionalDensityEstimator {
	private double p; // Scaling factor

	private double alpha, beta, gamma; // alpha = 4L^3/(wt); beta = w^4; gamma = t^4;

	private double muE, sigmaE, muX, sigmaX, muY, sigmaY;

	public LLCantilever(double L, double t, double w, double muE, double sigmaE, double muX, double sigmaX, double muY,
			double sigmaY) {
		p = 1.0 / 3.0;
		this.alpha = 4.0 * L * L * L / (w * t);
		this.beta = w * w * w * w;
		this.gamma = t * t * t * t;
		this.muE = muE;
		this.sigmaE = sigmaE;
		this.muX = muX;
		this.sigmaX = sigmaX;
		this.muY = muY;
		this.sigmaY = sigmaY;
	}

	public LLCantilever(double L, double t, double w, double muE, double sigmaE, double muX, double sigmaX, double muY,
			double sigmaY, double p) {
		this(L, t, w, muE, sigmaE, muX, sigmaX, muY, sigmaY);
		this.p = p;
	}

	@Override
	public double evalEstimator(double x, double[] data) {
		double E = data[0];
		double X = data[1];
		double Y = data[2];
		if ((alpha / E * Math.sqrt(X * X / beta + Y * Y / gamma)) > x)
			return 0;
		else
			return (  (- E * (p-1) * (E - muE)/(sigmaE*sigmaE)- X * p * (X - muX)/(sigmaX*sigmaX) - Y * p * (Y - muY)/(sigmaY*sigmaY) + 3.0 *p -1.0) / x  );
	}

	public double getP() {
		return p;
	}

	public void setP(double p) {
		this.p = p;
	}

}
