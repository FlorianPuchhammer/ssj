package umontreal.ssj.stat.density;

import java.io.FileWriter;
import java.io.IOException;

import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.ShortColumnVars;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.list.ListOfTallies;

public class LLShortColumn extends ConditionalDensityEstimator {

	double b, h;
	double muY, sigmaY;
	double p;

	public LLShortColumn(double b, double h, double muY, double sigmaY) {
		this.b = b;
		this.h = h;
		this.muY = transformMu(muY, sigmaY);
		this.sigmaY = transformSigma(muY, sigmaY);
		this.p = 0.5;
	}

	public LLShortColumn(double b, double h, double muY, double sigmaY, double p) {
		this.b = b;
		this.h = h;
		this.muY = transformMu(muY, sigmaY);
		this.sigmaY = transformSigma(muY, sigmaY);
		this.p = p;
	}

	private double transformMu(double mu, double sigma) {
		return (Math.log(mu) - 0.5 * Math.log(sigma * sigma / (mu * mu) + 1.0));
	}

	private double transformSigma(double mu, double sigma) {
		return (Math.sqrt(Math.log(1.0 + sigma * sigma / (mu * mu))));
	}

	private double g(double[] data) {
		return ( 1.0 - 4.0 * data[1] / (b * h * h * data[0]) - data[2] * data[2] / (b * b * h * h * data[0] * data[0]));
	}

	@Override
	public double evalEstimator(double x, double[] data) {
		if (g(data) >= x)
			return 0;
		if(data[2]<0)
			System.out.println("p<0");
		double fac1 = data[1] - 0.25 * b * h * h * data[0] ;
		double fac2 = (2.0 * (500.0 + data[2])-data[1] ) / 120000.0;
		double fac3 = Math.abs(data[2]) * (2000.0 + data[1] - 8.0 * data[2])/60000.0;
		return ( (fac1 * fac2) + 0.5* (fac3) + 0.5 + 1.0)/x ;
		
		
//		double val = ((data[1] - 0.25 * b * h * h * data[0]) * (2.0 * (500.0 + data[2]) - data[1]) / 120000.0
//				+ 0.5 * p * data[2] * (2000.0 + data[1] - 8.0 * data[2]) / 60000.0 - (1.0 - p) * 0.5 * data[0] * (muY - sigmaY*sigmaY - Math.log(data[0]))/(data[0] *sigmaY *sigmaY) + p + 0.5 )
//				/ x;
//		double val = ((data[1] - 0.25 * b * h * h * data[0]) * (2.0 * (500.0 + data[2]) - data[1]) / 120000.0
//				+ 0.5  * data[2] * (2000.0 + data[1] - 8.0 * data[2]) / 60000.0 + 1.5 )
//				/ x;
//		double val = ((data[1] - 0.25 * b * h * h * data[0]) * (2.0 * (500.0 + data[2]) - data[1]) / 120000.0
//				-  0.5 * data[0] * (muY - sigmaY*sigmaY - Math.log(data[0]))/(data[0] *sigmaY *sigmaY)  + 0.5 )
//				/ x;
//		return val;
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
		
		double pp = 0.0;
		ConditionalDensityEstimator cde = new LLShortColumn(bb,h,muY,sigmaY,pp);
		String descr = "LLShortColumn";
		
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
