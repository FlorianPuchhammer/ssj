package umontreal.ssj.stat.density;

import umontreal.ssj.stochprocess.BrownianMotionBridge;

public class LRLookBackOption extends ConditionalDensityEstimator {

	int dim; 

	double s0,  r,sigma, a, b, h, K;


	
	public LRLookBackOption(double a, double b, int dim, double s0, double K, double r, double sigma) {
		this.a = a;
		this.b = b;
		this.dim = dim;
		this.s0 = s0;
		this.K = K;
		this.sigma = sigma;
		this.r = r;
		this.h = 1.0 / (double) dim;

	}
	
	//note: data includes X(0)
	double g(double[] data) {
		double res = sigma * data[0];
		double temp;
		for(int j = 1; j <= dim; j++) {
			temp = (r- sigma*sigma*0.5) * h * j + sigma * data[j];
			if(res < temp)
				res = temp;
		}
		return s0*Math.exp(res);
	}
	
	@Override
	public double evalEstimator(double x, double[] data) {
		double res = 0.0;
		if (g(data) > x)
			res = -data[1]/(sigma * x * h);
		
		return res;
	}

}
