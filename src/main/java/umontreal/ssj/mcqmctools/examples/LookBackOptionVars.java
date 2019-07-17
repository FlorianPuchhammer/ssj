package umontreal.ssj.mcqmctools.examples;

import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stochprocess.BrownianMotion;
import umontreal.ssj.stochprocess.BrownianMotionBridge;

/**
 * Basically computes the path of a standard brownian motion sequentially.
 * @author puchhamf
 *
 */
public class LookBackOptionVars implements MonteCarloModelDoubleArray {

	int dim;
	double[] path;
	double h; //stepsize
	BrownianMotionBridge bm;
	
	
	public LookBackOptionVars(int dim, BrownianMotionBridge bm){
		this.dim = dim;
		path = new double[dim+1];
		this.h = 1.0/(double) dim;
		this.bm = bm;
		this.bm.setObservationTimes(h,dim);
	}
	
	public LookBackOptionVars(int dim){
		this(dim,new BrownianMotionBridge(0.0,0.0,1.0,new MRG32k3a()));
	}
	
	@Override
	public void simulate(RandomStream stream) {
//		double x = x0;
//	    path[0]=x0;
//	    for (int j = 0; j < dim; j++) {
//	        x += Math.sqrt (h) * NormalDist.inverseF01(stream.nextDouble());
//	        path[j + 1] = x;
//	    }
		
		bm.setStream(stream);
		path = bm.generatePath();

	}

	@Override
	public double[] getPerformance() {
		return path;
	}

	@Override
	public int getPerformanceDim() {
		return (dim+1);
	}

	public String toString() {
		return "Vars for lookback option with " + (dim) + " observations";
	}
}
