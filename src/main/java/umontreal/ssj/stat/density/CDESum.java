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
import umontreal.ssj.mcqmctools.examples.SumOfNormals;
import umontreal.ssj.mcqmctools.examples.SumOfNormalsArray;
import umontreal.ssj.probdist.ContinuousDistribution;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.randvar.NormalGen;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.list.ListOfTallies;
import umontreal.ssj.stochprocess.GeometricBrownianMotion;

public class CDESum extends ConditionalDensityEstimator {

	ContinuousDistribution dist;
	double norma;

	public CDESum(ContinuousDistribution dist, double norma) {
		this.dist = dist;
		this.norma = norma;
	}

	public CDESum(ContinuousDistribution dist) {
		this(dist, 1.0);
	}


	@Override
	public double evalEstimator(double x, double[] data) {
//		double arg = x * norma - data[0];
		return (dist.density(x * norma - data[0]) * norma);
	}

	public String toString() {
		return "CDESum";
	}
	
	public static void main(String[] args) throws IOException {
		int n = 32768 * 4;
		int mink = 15 + 2;
		int m = 1;
		int numEvalPoints = 100;
		RandomStream noise = new MRG32k3a();
		
		int dim = 10;
		double[] mu = new double[dim];
		Arrays.fill(mu,0.0);
		double[] sigma = new double[dim];
		sigma[0] = 1.0;
		for(int j = 1; j < dim; ++j)
			sigma[j] = sigma[j-1]  * 2.0;
		
		double norma = 0.0;
		double sigmaHidden = sigma[dim-1] * 2.0; //last one
		for(double s:sigma)
			norma += s*s;
		norma += sigmaHidden * sigmaHidden;
		norma = Math.sqrt(norma);
		
		SumOfNormalsArray model = new SumOfNormalsArray(mu,sigma);
		

		
		String outdir = "SumOfNormals";
		
	
		double a = -2.0; double b=2.0; 
//		a-=norma;
//		b+=norma;
//		norma=1.0;
	
		ConditionalDensityEstimator cde = new CDESum(new NormalDist(0.0,sigmaHidden),norma);
		String descr = "CDESumOfNormals";
		
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
