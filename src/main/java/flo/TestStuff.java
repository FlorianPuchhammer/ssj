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
import umontreal.ssj.hups.IndependentPointsCached;
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
import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.BucklingStrengthVars;
import umontreal.ssj.mcqmctools.examples.MultiNormalIndependent;
import umontreal.ssj.mcqmctools.examples.SumOfNormals;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.PgfDataTable;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.density.CDEBucklingStrength;
import umontreal.ssj.stat.density.CDECantilever;
import umontreal.ssj.stat.density.CDENormalizedSumOfNormals;
import umontreal.ssj.stat.density.ConditionalDensityEstimator;
import umontreal.ssj.stat.density.DEDerivativeGaussian;
import umontreal.ssj.stat.density.DensityEstimator;
import umontreal.ssj.stat.list.ListOfTallies;
import umontreal.ssj.util.sort.BatchSort;
import umontreal.ssj.util.sort.HilbertCurveBatchSort;
import umontreal.ssj.util.sort.HilbertCurveSort;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.SplitSort;

public class TestStuff {
	private static double[] genEvalPoints(int numPts, double a, double b, RandomStream stream) {
		double[] evalPts = new double[numPts];
		double invNumPts = 1.0 / ((double) numPts);
		for (int i = 0; i < numPts; i++)
			evalPts[i] = a + (b - a) * ((double) i + stream.nextDouble()) * invNumPts;
		return evalPts;
	}
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

		
		RandomStream rand = new MRG32k3a();
		double [] data = new double[585];
		for(int j=0; j< data.length;j++)
			data[j] = rand.nextDouble();
		Arrays.sort(data);
		
		double[] evalPts = new double[200];
		for(int j=1; j< evalPts.length; j++)
			evalPts[j] = evalPts[j-1] + 1.0/(double)evalPts.length;
		
		DensityEstimator de = new DEDerivativeGaussian(1.0/32.0, data);
		
		((DEDerivativeGaussian) de).evalDensity(evalPts);
		de.evalDensity(evalPts);
			
		
		System.out.println("A -- O K ! ! !");
	}

}
