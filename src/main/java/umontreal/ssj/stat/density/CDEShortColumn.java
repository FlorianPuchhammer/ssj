package umontreal.ssj.stat.density;

import java.io.FileWriter;
import java.io.IOException;

import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.SanVarsCDE;
import umontreal.ssj.mcqmctools.examples.ShortColumn;
import umontreal.ssj.mcqmctools.examples.ShortColumnVars;
import umontreal.ssj.probdist.LognormalDist;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.list.ListOfTallies;

public class CDEShortColumn extends ConditionalDensityEstimator {

	double b, h;
	double muY, sigmaY;

	public CDEShortColumn(double b, double h, double muY, double sigmaY) {
		this.b = b;
		this.h = h;
		this.muY = transformMu(muY, sigmaY);
		this.sigmaY = transformSigma(muY, sigmaY);
	}

	private double transformMu(double mu, double sigma) {
		return (Math.log(mu) - 0.5 * Math.log(sigma * sigma / (mu * mu) + 1.0));
	}

	private double transformSigma(double mu, double sigma) {
		return (Math.sqrt(Math.log(1.0 + sigma * sigma / (mu * mu))));
	}

	@Override
	public double evalEstimator(double x, double[] data) {
//		double 1mX = 1.0 - x;
		//Sqrt[ 4 M^2 +( P^2 h^2 (1 -z)) ]
		double temp1 = Math.sqrt(4.0 * data[1] * data[1] + (data[2] * data[2] * h * h * (1.0 - x)));
		double arg = (2.0 * data[1] + temp1)/( b * h * h * (1.0 - x));
		
		double val = LognormalDist.density(muY, sigmaY, arg);
		val *=  (arg/(1.0-x) - 0.5 * data[2] * data[2] / (b * (1.0-x) * temp1) );
		return val;
	}

	public static void main(String[] args) throws IOException {
		int n = 32768 * 4;
		int mink = 15 + 2;
		int m = 1;
		int numEvalPoints = 100;
		RandomStream noise = new MRG32k3a();
		
		double h = 15.0;
		double bb = 5.0;
		double muY = 5.0;
		double muM = 2000.0;
		double muP = 500.0;
		double sigmaY = 0.5;
//		double[][] sigma = {{400.0, 100.0}, {100.0, 100.0}};
//		double [][] sigma = {{160000.0, 20000.0}, {20000.0, 10000.0}};
		double[][] sigma = {{400.0, 50.0}, {0.0, 86.60254037844386}};
//		double [][] sigma = {{400.0, 50.0}, {0.0, 50.0 * Math.sqrt(3.0)}};
		MonteCarloModelDoubleArray model = new ShortColumnVars(muY,muM,muP,sigmaY,sigma);

		int dim = 3;
		
		double a =  -5.338; //ShortColumn
		double b = -0.528;
		
		double [] evalPoints = genEvalPoints(numEvalPoints,a,b); 
		
		ConditionalDensityEstimator cde = new CDEShortColumn(bb,h,muY,sigmaY);
		String descr = "CDEShortColumn";
		
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
