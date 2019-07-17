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
import umontreal.ssj.mcqmctools.examples.BucklingStrengthVars;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.list.ListOfTallies;

public class LLBucklingStrength extends ConditionalDensityEstimator {

	double mud, mue; 
	double sigmad, sigmae;
	double [] weights;
	
	public LLBucklingStrength(double mud, double sigmad, double mue, double sigmae,double[] weights) {
		this.mue = mue;
		this.sigmae = sigmae;
		this.mud = mud;
		this.sigmad = sigmad;
		this.weights = weights;
	}
	
	private double strength(double[] data) {
		return ((2.1 / data[1] - 0.9 / (data[1] * data[1])) * (1.0 - 0.75 * data[2] / data[1])
				* (1.0 - data[0] * data[3]));
	}

	@Override
	public double evalEstimator(double x, double[] data) {
		if (strength(data) > x)
			return 0;
		else {
		
			return ( weights[0] * (-4.0  * data[1]/3.0 + data[2]) * (mud  - data[2])/(sigmad * sigmad) + weights[1] * (data[3] - 1.0/data[0]) * (mue - data[3])/(sigmae * sigmae)+1 ) / x;
		}
	}
	
	public String toString() {
		return "LLBucklingStrength";
	}

	public double[] getWeights() {
		return weights;
	}

	public void setWeights(double [] weights) {
		this.weights = weights;
	}

	public static void main(String[] args) throws IOException {
		int n = 32768 * 4;
		int mink = 15 + 2;
		int m = 1;
		int numEvalPoints = 100;
		RandomStream noise = new MRG32k3a();
		
		double[] mus = { 0.992 * 24.0, 1.05 * 0.5, 1.3 * 34.0, 0.987 * 29.0E3,0.35, 5.25};
		double[] covs = { 0.028, 0.044, 0.1235, 0.076,0.05,0.07 };
		int dim = mus.length;
		double[] sigmas = new double[dim];
		for(int j = 0; j < dim; j++)
			sigmas[j] = mus[j] * covs[j];
		MonteCarloModelDoubleArray model = new BucklingStrengthVars(mus, sigmas);
		
		double a = 0.5169;
		double b = 0.6511;
		double[] weights = {0.5,0.5};
		
		double [] evalPoints = genEvalPoints(numEvalPoints,a,b); 
		
		ConditionalDensityEstimator cde = new LLBucklingStrength(mus[4],sigmas[4],mus[5],sigmas[5],weights);
		String descr = "LLBucklingStrength";
		
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
