package flo;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import flo.biologyArrayRQMC.examples.ChemicalReactionNetwork;
import flo.biologyArrayRQMC.examples.SchloeglSystemProjected;
import flo.biologyArrayRQMC.examples.SchloeglSystemProjectedSort;
import umontreal.ssj.functionfit.LeastSquares;
import umontreal.ssj.functionfit.SmoothingCubicSpline;
import umontreal.ssj.hups.BakerTransformedPointSet;
import umontreal.ssj.hups.CachedPointSet;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.NestedUniformScrambling;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetIterator;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.RandomShift;
import umontreal.ssj.hups.Rank1Lattice;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.hups.SortedAndCutPointSet;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChains;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.PgfDataTable;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.density.DensityEstimator;
import umontreal.ssj.util.sort.BatchSort;
import umontreal.ssj.util.sort.HilbertCurveBatchSort;
import umontreal.ssj.util.sort.HilbertCurveSort;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.SplitSort;

public class plotPaths {

	private static double coefficientOfDetermination(double[] data, double[] dataEstimated) {
		int i;
		int max = data.length;
		double maxInv = 1.0 / (double) max;
		double dataMean = 0.0;
		double SSres = 0.0;
		double SStot = 0.0;
		for (i = 0; i < max; i++)
			dataMean += data[i];
		dataMean *= maxInv;
		for (i = 0; i < max; i++) {
			SSres += (data[i] - dataEstimated[i]) * (data[i] - dataEstimated[i]);
			SStot += (data[i] - dataMean) * (data[i] - dataMean);
		}
		return 1.0 - SSres / SStot;
	}

	private static double l2Dist(double[] x, double[] y) {
		double temp;
		double dist = 0.0;
		for (int i = 0; i < x.length; ++i) {
			temp = x[i] - y[i];
			dist += temp * temp;
		}
		return Math.sqrt(dist);
	}

	private static double l1Dist(double[] x, double[] y) {

		double dist = 0.0;
		for (int i = 0; i < x.length; ++i) {
			dist += Math.abs(x[i] - y[i]);
		}
		return dist;
	}

	private static String printTuples(int[][] tuples) {
		int rows = tuples.length;
		int cols = tuples[0].length;
		String str = "{ {" + tuples[0][0];
		for (int j = 1; j < cols; ++j)
			str += "," + tuples[0][j];
		str += "}";

		for (int i = 1; i < rows; ++i) {
			str += ", {" + tuples[i][0];
			cols = tuples[i].length;
			for (int j = 1; j < cols; ++j)
				str += "," + tuples[i][j];
			str += "}";
		}
		str += " }";
		return str;
	}

	private static double correlation(double[] xs, double[] ys) {
		// TODO: check here that arrays are not null, of the same length etc

		double sx = 0.0;
		double sy = 0.0;
		double sxx = 0.0;
		double syy = 0.0;
		double sxy = 0.0;

		int n = xs.length;
		double nInv = 1.0 / (double) n;

		for (int i = 0; i < n; ++i) {
			double x = xs[i];
			double y = ys[i];

			sx += x;
			sy += y;
			sxx += x * x;
			syy += y * y;
			sxy += x * y;
		}

		// cov
		double cov = sxy - sx * sy * nInv;
		// standard error of x
		double sigmax = Math.sqrt(sxx - sx * sx * nInv);
		// standard error of y
		double sigmay = Math.sqrt(syy - sy * sy * nInv);

		// correlation is just a normalized cov
		return cov / sigmax / sigmay;
	}

	public static void main(String[] args) throws IOException {

		String filepath = "/u/puchhamf/misc/jars/chemical/schloeglProjected/data/";
		String filename = "MCData_Step_";

		int numPaths = 30;
		int numSteps = 15;
		int numCols = 3;
		int pre = 2000;
		
		double [][] paths = new double [numPaths][numSteps+1];
		

		
		Scanner sc;
		
		String[] line;
		
		for(int s = 0; s < numSteps; s++) {
			sc  = new Scanner(new BufferedReader(new FileReader(filepath + filename + s + ".csv")));
			for(int r =0; r < pre; ++r)
				sc.nextLine();
			for(int p = 0; p < numPaths; p++) {
				line = sc.nextLine().trim().split(",");
				paths[p][s] = Double.parseDouble(line[0]);
				if(s== (numSteps-1))
					paths[p][numSteps] = Double.parseDouble(line[numCols-1]);
			}
		}
		
		
	
		
			
		ArrayList<PgfDataTable> pgfTblList = new ArrayList<PgfDataTable>();
		double[][] plotData;
		for (int p = 0; p < numPaths; ++p) {
			 plotData= new double[numSteps+1][];
			for(int s = 0; s <= numSteps ; s++)	{
				plotData[s] = new double[2];
				plotData[s][0] =s;
				plotData[s][1] = paths[p][s];
			}
				pgfTblList.add(new PgfDataTable("tblName", "", new String[] {"var1", "var2"},plotData));
		}

		StringBuffer sb = new StringBuffer("");
		
			sb.append(PgfDataTable.pgfplotFileHeader());
			sb.append(PgfDataTable.drawPgfPlotManyCurves("title", "axis", 0, 1, pgfTblList, 2,
					"", " "));
			sb.append(PgfDataTable.pgfplotEndDocument());
			
			FileWriter fw = new FileWriter(filepath  + "paths.tex");

			fw.write(sb.toString());
			fw.close();

			
		
		System.out.println("A -- O K ! ! !");
	}

}
