package umontreal.ssj.mcqmctools.examples;

import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stochprocess.BrownianMotion;
import umontreal.ssj.stochprocess.BrownianMotionBridge;

public class LookBackOption implements MonteCarloModelDouble {

	int dim;
	double[] path;
	
	double s0,K,r,sigma,h;
	
	BrownianMotionBridge bm;
	
	public LookBackOption(int dim, double s0, double K, double r, double sigma, BrownianMotionBridge bm) {
		this.dim = dim;
		this.s0=s0;
		this.K=K;
		this.r=r;
		this.sigma=sigma;
		
		
		this.h = 1.0/(double) dim;
		this.path = new double [dim+1];
		this.bm=bm;
		this.bm.setObservationTimes(h,dim);
	}
	
	public LookBackOption(int dim, double s0, double K, double r, double sigma) {
		this(dim,s0,K,r,sigma,new BrownianMotionBridge(0.0,0.0,1.0,new MRG32k3a()));
	}
	
	@Override
	public void simulate(RandomStream stream) {
		bm.setStream(stream);
		path = bm.generatePath();
		
		
	}

	@Override
	public double getPerformance() {
		
		double res = sigma * path[0];
		double temp;
		for(int j = 1; j <= dim; j++ ) {
			temp = (r- sigma*sigma*0.5) * h * j + sigma * path[j];
			if(res < temp)
				res = temp;
		}

		return s0 * Math.exp(res);
	}

	public String toString() {
		return "Lookback Option with " + dim + "observations.";
	}
}
