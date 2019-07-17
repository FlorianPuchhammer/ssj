package umontreal.ssj.stat.density;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.AsianOptionVars;
import umontreal.ssj.mcqmctools.examples.ShortColumnVars;
import umontreal.ssj.probdist.LognormalDist;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.randvar.NormalGen;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.list.ListOfTallies;
import umontreal.ssj.stochprocess.BrownianMotion;
import umontreal.ssj.stochprocess.GeometricBrownianMotion;

public class CDEAsianOption extends ConditionalDensityEstimator {

	int dim; // simulate dim, estimate for dim+1

	double K, discount, s0, x0, sigma, r, a, b;
	double[] weights;
	

	public CDEAsianOption(double a, double b, int dim, double s0, double K, double sigma, double r) {
		this.a = a;
		this.b = b;
		this.dim = dim;
		this.s0 = s0;
		this.r = r;
		this.sigma = sigma;
		this.K = K;
		
		this.weights = new double[dim];
		double dimInv = 1.0/(double) dim;
		Arrays.fill(weights,dimInv);

	}
	
	public CDEAsianOption(double a, double b, int dim, double s0, double K, double sigma, double r, double [] weights) {
		this.a = a;
		this.b = b;
		this.dim = dim;
		this.s0 = s0;
		this.r = r;
		this.sigma = sigma;
		this.K = K;
		
		this.weights = weights;

	}

	public double evalEstimator(double x, double[] data) {
		double val = 0.0; // value to return
		double sum; // sum of all vars except for one
		double temp, tk;
		double rho = r - sigma * sigma *0.5;

		for (int k = 0; k < dim; ++k) { // k is the variable that is left out in "sum"
			sum = 0.0;
			for (int j = 0; j < dim; ++j) // add al vars except for k
				if (j != k)
					sum += data[j];
			tk = ((double) (k + 1.0)) / ((double) dim);
			temp = 1.0 / (s0 * Math.exp(rho * tk));
			val += LognormalDist.density(0, sigma * Math.sqrt(tk), (((double) dim) * (x +K) - sum) * temp) * temp * (double) dim * weights[k];
//			val += LognormalDist.density(0, sigma * tk, (((double) dim) * (x + K) - sum) * temp) * temp * (double) dim * weights[k];

//			val += NormalDist.density(0, Math.sqrt((double) (k + 1) / (double) dim),
//					(k + 1) / (sigma * dim)
//							* (-(r - sigma * sigma * 0.5) + 1 / sigma * Math.log((dim * (x + K) - sum) / (double) s0)))
//					* (double) dim / ((dim * (x + K) - (dim - 1) * (sum)) * sigma);

		}
		// System.out.println("val"+val);
//		val /= (double) dim; // this has been considered in val already. Each summand would be multiplied dim!!!
		return val;

	}
	
	public static void main(String[] args) throws IOException {
		int n = 32768 * 4;
		int mink = 15 + 2;
		int m = 1;
		int numEvalPoints = 100;
		RandomStream noise = new MRG32k3a();
		
		double strike= 101.0;
		double s0 = 100.0;
		double sigma = 0.12136;
		int dim =12;
		double[] obsTimes = new double[dim + 1];
		obsTimes[0] = 0.0;
		for (int j = 1; j <= dim; j++) {
			obsTimes[j] = (double) j / (double) dim;
		}
		double r =Math.log(1.09); r = 0;
		
		AsianOptionVars model = new AsianOptionVars(r,dim, obsTimes,strike);
		NormalGen gen = new NormalGen(new MRG32k3a());
//		GeometricBrownianMotion sp = new GeometricBrownianMotion(s0, 0.1,
//				sigma, new BrownianMotion(0.0, 0.0, 1.0, gen));
		GeometricBrownianMotion sp = new GeometricBrownianMotion(s0, 0.1,
				sigma,new MRG32k3a());
		model.setProcess(sp);
		
		double a = 0.0; double b=27.13; 
		double [] weights = new double[dim];
		double norma = 0.0;
		for(int j = 0; j < dim; j++) {
//			weights[j] = 1.0/(double)((j+1.0)*(j+1.0));
			weights[j] = (double)((j+1.0));
			norma += weights[j];
		}
		for(int j = 0; j < dim; j++) {
			weights[j] /= norma;
//			System.out.println(j + ", " + weights[j]);
		}
		
		ConditionalDensityEstimator cde = new CDEAsianOption(a,b, dim,
				s0, strike,sigma,r,weights);
		String descr = "CDEAsianOption";
		
		double [] evalPoints = genEvalPoints(numEvalPoints,a,b); 
		
		
		
		PointSet p = new SobolSequence( mink, 31,dim);

		PointSetRandomization rand = new LMScrambleShift(noise);
		double[][][] data = new double[m][n][model.getPerformanceDim()];
		ListOfTallies<Tally> statRepsList = ListOfTallies.createWithTally(model.getPerformanceDim());

		RQMCExperiment.simulReplicatesRQMC(model, new RQMCPointSet(p,rand), m, statRepsList, data);
		
		double[] density = new double[numEvalPoints];
		density = cde.evalDensity(evalPoints, data[0]);
		
		String[] axis = {"x", "dens"};
		FileWriter fw = new FileWriter(descr + "_density.tex");
		fw.write(		plotDensity(evalPoints,density,descr,axis));
		fw.close();
		
	}


}
