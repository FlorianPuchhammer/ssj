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
import umontreal.ssj.mcqmctools.examples.SanVars;
import umontreal.ssj.mcqmctools.examples.SanVarsCDE;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.list.ListOfTallies;

public class LLSan extends ConditionalDensityEstimator {
	double paths[];
	
	public LLSan() {
		paths = new double[6];
	}
	private double computePaths(double[] data) {
		double[] paths = new double[6];
		double maxPath;
		paths[0] = data[0] + data[2] + data[5] + data[10];
		paths[1] = data[0] + data[3] + data[6] + data[11] + data[12];
		paths[2] = data[0] + data[3] + data[7] + data[8] + data[12];
		paths[3] = data[0] + data[3] + data[7] + data[9] + data[10];
		paths[4] = data[0] + data[4] + data[10];
		paths[5] = data[1] + data[5] + data[10];
		maxPath = paths[0];
		for (int p = 1; p < 6; p++)
			if (paths[p] > maxPath)
				maxPath = paths[p];
		return maxPath;
	}

	@Override
	public double evalEstimator(double x, double[] data) {
		if (computePaths(data) > x)
			return 0.0;
		return ((13.0 - data[0]) / (3.25 * 3.25) * data[0] + (5.5 - data[1]) / (1.375 * 1.375) * data[1] - data[2] / 7.0
				+ (5.2 - data[3]) / (1.3 * 1.3) * data[3] - data[4] / 16.5 - data[5] / 14.7 - data[6] / 10.3
				- data[7] / 6.0 - data[8] * 0.25 - data[9] * 0.05 + (3.2 - data[10]) / (0.8 * 0.8) * data[10]
				+ (3.2 - data[11]) / (0.8 * 0.8) * data[11] - data[12]/16.5 + 13.0) / x;
	}
	
	@Override
	public String toString() {
		return "LLSan13";
	}
	public static void main(String[] args) throws IOException {
		int n = 32768*4;
		int mink = 15+2;
		int m = 1;
		int numEvalPoints = 100;
		RandomStream noise = new MRG32k3a();
		
		MonteCarloModelDoubleArray model = new SanVars("san13a.dat"); //CDE
		int dim = 13;
		
		double a =22.0;
		double b = 106.24; //95% non-centralized
		
		double [] evalPoints = genEvalPoints(numEvalPoints,a,b); 
		
		ConditionalDensityEstimator cde = new LLSan();
		String descr = "LLSan";
		
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
