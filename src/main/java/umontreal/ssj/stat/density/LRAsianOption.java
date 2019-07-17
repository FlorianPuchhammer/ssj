package umontreal.ssj.stat.density;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.AsianOptionVars;
import umontreal.ssj.randvar.NormalGen;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.list.ListOfTallies;
import umontreal.ssj.stochprocess.GeometricBrownianMotion;

//Needs input payoff+K, so sum S_j's
public class LRAsianOption extends ConditionalDensityEstimator {

	int dim; // simulate dim, estimate for dim+1

	double K, discount, s0, x0, sigma, r, a, b;

	public LRAsianOption(double a, double b, int dim, double s0, double K, double sigma, double r) {
		this.a = a;
		this.b = b;
		this.dim = dim;
		this.s0 = s0;
		this.r = r;
		this.sigma = sigma;
		this.K = K;

	}

	public double sToB(double S, double t) {
		double rho = r - sigma * sigma * 0.5;
		return ((Math.log(S / s0) - rho * t) / sigma);
	}

	public double g(double[] data) {
		double sum = 0.0;
		for (int j = 0; j < data.length; j++)
			sum += data[j];
		return (sum / (double) dim);
	}

	@Override
	public double evalEstimator(double x, double[] data) {
		if (g(data) >= x)
			return 0;
		
		double rho = r - sigma * sigma * 0.5;
		double tj;
		double sum ;
		double [] B = new double[dim];
		for(int j = 0; j < dim; ++j) {
			tj = (double)(j+1.0)/(double) dim;
			B[j] = sToB(data[j] ,tj);
		}
		
		sum = 12.0 * B[0] * (-2.0 * B[0] + B[1]);
		sum += 12.0 *B[dim-1] * (-B[dim-1] + B[dim -2]);
		sum += dim;
		
		for(int j = 1; j < dim-1; ++j)
			sum += 12.0 * B[j] * (B[j-1] - 2.0 * B[j] + B[j+1]);

		return ( (1.0-1.0/sigma) * (sum)/(x * Math.log(x))  );
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
		
		double a = 0.0 + strike; double b=27.13 + strike; 
		double [] weights = new double[dim];
		double norma = 0.0;
		
		
		
		ConditionalDensityEstimator cde = new LRAsianOption(a,b, dim,
				s0, strike,sigma,r);
		String descr = "LRAsianOption";
		
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
