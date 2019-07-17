package umontreal.ssj.stat.density;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.BucklingStrengthVars;
import umontreal.ssj.mcqmctools.examples.SanVarsCDE;
import umontreal.ssj.probdist.ContinuousDistribution;
import umontreal.ssj.probdist.DistributionFactory;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.list.ListOfTallies;

public class CDESan extends ConditionalDensityEstimator {

public ContinuousDistribution[] dist = new ContinuousDistribution[5];
	
	public CDESan(String fileName) throws IOException  {
		readDistributions(fileName);
		
	}

	private void readDistributions(String fileName) throws IOException {
		// Reads data and construct arrays.
		BufferedReader input = new BufferedReader(new FileReader(fileName));
		Scanner scan = new Scanner(input);
		int i = 0;
		for (int k = 0; k < 13; k++) {
			
			if(k == 4 || k == 5 || k == 6 || k == 8 || k == 9){
			dist[i] = DistributionFactory.getContinuousDistribution(scan
					.nextLine());
//			System.out.println("distr. " + i + " is " + dist[i].toString());
			i++;
			}
			else
				scan.nextLine();
		}
		scan.close();
	}
	@Override
	public double evalEstimator(double x, double[] data) {
		int t = data.length;
		double dens = 0.0;
		for(int j = 0; j < t; j++){
			double prod = 1.0;
			for(int k = 0; k< t; k++){
				if(k != j)
//					System.out.println("Zeigs:" + data[k]);
					prod *= dist[k].cdf(x - data[k]);
			}
			dens += dist[j].density(x-data[j]) * prod;
		}
			
		return dens;
	}

	public String toString(){
		return "CDESan13";
	}

	public static void main(String[] args) throws IOException {
		int n = 32768 * 4;
		int mink = 15 + 2;
		int m = 1;
		int numEvalPoints = 100;
		RandomStream noise = new MRG32k3a();
		
		MonteCarloModelDoubleArray model = new SanVarsCDE("san13a.dat"); //CDE
		int dim = 13;
		
		double a =22.0;
		double b = 106.24; //95% non-centralized
		
		double [] evalPoints = genEvalPoints(numEvalPoints,a,b); 
		
		ConditionalDensityEstimator cde = new CDESan("san13a.dat");
		String descr = "CDESan";
		
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
