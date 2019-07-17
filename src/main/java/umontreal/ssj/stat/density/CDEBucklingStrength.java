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
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.list.ListOfTallies;

public class CDEBucklingStrength extends ConditionalDensityEstimator {

	double mud, mue;
	double sigmad, sigmae;
	double p;

	public CDEBucklingStrength(double mud, double sigmad, double mue, double sigmae, double p) {
		this.mue = mue;
		this.sigmae = sigmae;
		this.mud = mud;
		this.sigmad = sigmad;
		this.p = p;
	}

	@Override
	public double evalEstimator(double x, double[] data) {

		double fac1 = 2.1 / data[1] - 0.9 / (data[1] * data[1]);
		double fac2 = 1.0 - data[0] * data[3];
		double fac3 = 1.0 - 0.75 * data[2] / data[1];
		double fac4 = data[1] / (0.75);
		double fac5 = 1.0 / (data[0]);

		if (fac1 < 0 || fac2 < 0 || fac3 < 0 || fac4 < 0 || fac5 < 0) {
			System.out.println(fac1 + "\t" + fac2 + "\t" + fac3 + "\t" + fac4 + "\t" + fac5);
			return 0.0;
		}
		// System.out.println("Dens:" +( p * NormalDist.density(mud, sigmad, (1.0 - x *
		// fac4)) * fac4
		// ));
		else
			return (p * NormalDist.density(mud, sigmad, (1.0 - x / (fac1 * fac2)) * fac4) * fac4 / (fac1 * fac2)
					+ (1.0 - p) * NormalDist.density(mue, sigmae, (1.0 - x / (fac1 * fac3)) * fac5) * fac5
							/ (fac1 * fac3));
	}

	public String toString() {
		return "CDEBucklingStrength";
	}

	public double getP() {
		return p;
	}

	public void setP(double p) {
		this.p = p;
	}
	
	public static void main(String[] args) throws IOException {
		int n = 32768;
		int mink = 15;
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
		double pp = 0.0;
		
		double [] evalPoints = genEvalPoints(numEvalPoints,a,b); 
		
		ConditionalDensityEstimator cde = new CDEBucklingStrength(mus[4],sigmas[4],mus[5],sigmas[5],pp);
		String descr = "CDEBucklingStrength";
		
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
