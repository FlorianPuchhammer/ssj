package flo;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import umontreal.ssj.hups.IndependentPointsCached;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.RandomShift;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.MultiNormalIndependent;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.PgfDataTable;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.density.CDECantilever;
import umontreal.ssj.stat.density.ConditionalDensityEstimator;
import umontreal.ssj.stat.list.ListOfTallies;

public class PlotCDEMCvsRQMC {
	public static void main(String[] args) throws IOException {

		int n = 128 *8; //n = 65536;
		int mink = 7 +3; //mink = 16;
		int m = 1;
		int numEvalPoints = 100;
		RandomStream noise = new MRG32k3a();
		
//		double a = -2;
//		double b = 2.0;
//		
//		
//		int dim = 5;
//		double[] mu = new double[dim];
//		Arrays.fill(mu,0.0);
//		double[] sigma = new double[dim];
//		
//		Arrays.fill(sigma,1.0);
//		
////		sigma[0] = 1.0;
////		for(int j = 1; j<dim; j++)
////			sigma[j] = sigma[j-1] / Math.sqrt(2.0);
//		
////		double norma = Math.sqrt((double)dim);
//		
//		
//		MonteCarloModelDoubleArray model = new MultiNormalIndependent(mu,sigma);
//		
//		double[] weights = new double[dim];
////		Arrays.fill(weights,1.0/((double) dim)); //set the unused ones negative, then they will be omitted!	
//		Arrays.fill(weights,-1.0); //set the unused ones negative, then they will be omitted!
//		weights[dim-1] = 1.0;
//		
//		ConditionalDensityEstimator cde = new CDENormalizedSumOfNormals(dim,weights,sigma);
	
		// BUCKLING
//		double[] mus = { 0.992 * 24.0, 1.05 * 0.5, 1.3 * 34.0, 0.987 * 29.0E3,0.35, 5.25};
//		double[] covs = { 0.028, 0.044, 0.1235, 0.076,0.05,0.07 };
//		int dim = mus.length;
//		double[] sigmas = new double[dim];
//		for(int j = 0; j < dim; j++)
//			sigmas[j] = mus[j] * covs[j];
//		MonteCarloModelDoubleArray model = new BucklingStrengthVars(mus, sigmas);
//		
//		double a = 0.5169;
//		double b = 0.6511;	
//
//		double pp = 1.0; //CDE guess
////		double pp = 0.00244140625; //CDE opt
////		double pp = 0.002378784944706333; // CDE quad fit
//
//		ConditionalDensityEstimator cde = new CDEBucklingStrength(mus[4],sigmas[4],mus[5],sigmas[5],pp);
//		String descr = "CDEBucklingStrengthG5";
		


		// CANTI, NORMALS
		double[] mus = {2.9E7,500.0,1000.0}; //Canti
		double[] sigmas = {1.45E6,100.0,100.0}; //Canti
		int dim = mus.length;
		
		MonteCarloModelDoubleArray model = new MultiNormalIndependent(mus,sigmas);
		
		double a = 0.407;
		double b = 1.515;	
		double D0 = 2.2535;
		a = (a+1) * D0;
		b = (b+1) * D0;
		a=4.0;
		b=4.9;
		
		double L = 100.0;
		double t = 2.0;
		double w = 4.0;
//		double[] weights = {0.25639625873702715, 8.639295478835745E-7 , 0.743602877333425};
//		double[] weights = {0.25, -1.0 , 0.75};
		double[] weights = {1.0, -1.0,-1.0};

		ConditionalDensityEstimator cde = new CDECantilever(L, t, w, mus[0], sigmas[0], mus[1], sigmas[1], mus[2],
				sigmas[2], weights);
		String descr = "cdeCantiX";
		
		double[] evalPoints = new double[numEvalPoints];
		evalPoints = genEvalPoints(numEvalPoints,a,b,noise);

		

		
		
		
		
		RQMCPointSet[] rqmc = new RQMCPointSet[2];
		PointSet p;
		PointSetRandomization rand;
		double[][][] data = new double[m][n][model.getPerformanceDim()];
		
		p = new IndependentPointsCached(n,dim);
		rand = new RandomShift(noise);
		rqmc[0] =  new RQMCPointSet(p,rand);
		rqmc[0].setLabel("MC");
		
		
		 p = new SobolSequence( mink, 31,dim);
		 rand = new LMScrambleShift(noise);
		rqmc[1] =  new RQMCPointSet(p,rand);
		rqmc[1].setLabel("RQMC");
		
		ListOfTallies<Tally> statRepsList;
		ArrayList<PgfDataTable> pgfTblList = new ArrayList<PgfDataTable>();
		double[] dens;
		double[][] plotData;
		
		for(RQMCPointSet prqmc : rqmc) {
			 statRepsList = ListOfTallies.createWithTally(model.getPerformanceDim());
			RQMCExperiment.simulReplicatesRQMC(model,prqmc , m, statRepsList, data);
			dens = new double[numEvalPoints];
			dens = cde.evalDensity(evalPoints, data[m-1]);
			plotData = new double[numEvalPoints][2];
			for(int s = 0; s < numEvalPoints ; s++)	{
				plotData[s][0] =evalPoints[s];
				plotData[s][1] = dens[s];
			}
			pgfTblList.add(new PgfDataTable("tblName", prqmc.getLabel(), new String[] {"var1", "var2"},plotData));
			
		}
		
//		Arrays.fill(weights,1.0/(double) dim);
//		cde = new CDENormalizedSumOfNormals(dim,weights,sigma);
	
		n*=16;
		mink += 4;
		 p = new SobolSequence( mink, 31,dim);
		 rand = new LMScrambleShift(noise);
		data = new double[m][n][model.getPerformanceDim()];
		 statRepsList = ListOfTallies.createWithTally(model.getPerformanceDim());
			RQMCExperiment.simulReplicatesRQMC(model,new RQMCPointSet(p,rand) , m, statRepsList, data);
			dens = new double[numEvalPoints];
			dens = cde.evalDensity(evalPoints, data[m-1]);
			plotData = new double[numEvalPoints][2];
			for(int s = 0; s < numEvalPoints ; s++)	{
				plotData[s][0] =evalPoints[s];
				plotData[s][1] = dens[s];
			}
			pgfTblList.add(new PgfDataTable("tblName","trueDens", new String[] {"var1", "var2"},plotData));
	

//		plotData = new double[numEvalPoints][2];
//		for(int s = 0; s < numEvalPoints ; s++)	{
//			plotData[s][0] =evalPoints[s];
//			plotData[s][1] = NormalDist.density01(evalPoints[s]);
//		}
//		pgfTblList.add(new PgfDataTable("tblName","trueDens", new String[] {"var1", "var2"},plotData));
		

		StringBuffer sb = new StringBuffer("");
		
			sb.append(PgfDataTable.pgfplotFileHeader());
			sb.append(PgfDataTable.drawPgfPlotManyCurves("title", "axis", 0, 1, pgfTblList, 2,
					"", " "));
			sb.append(PgfDataTable.pgfplotEndDocument());
			
			FileWriter fw = new FileWriter("mcVSrqmc_"  + "SumOfNormals.tex");

			fw.write(sb.toString());
			fw.close();

			
		
		System.out.println("A -- O K ! ! !");
	}
}
