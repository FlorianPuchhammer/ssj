package flo;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Scanner;

import flo.biologyArrayRQMC.examples.PKASort;
import umontreal.ssj.functionfit.LeastSquares;

public class TestVariableImportance {

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

	public static void main(String[] args) throws FileNotFoundException {

		int numSteps = 20;
		int rows = 262144; 	 rows *=4;
//		int rows = 128;
//		int cols = 6; //PKA 
//		int cols = 4; // EnzymeKinetics
//		 int cols = 2; //Schloegl
		 int cols = 11; //MAPK

		int colsExtended = 0 
				+ cols 
				+ (cols * (cols + 1)) / 2
//				+ (cols * (cols + 1) * (cols + 2)) / 6
		;

		Scanner sc;
		double[][] vars = new double[rows][cols];
		double[] response = new double[rows];
		double[][] varsExtended = new double[rows][colsExtended];
		double[] reg;
		int index;
		int[][] tuples = new int[colsExtended][];

		// read vars and response
		// for (int s = 1; s < numSteps; s++) {
		String perm = "data/PKA/permutation1M2.dat";
//		sc = new Scanner(new BufferedReader(new FileReader("data/MAPK6/MCDataLessNoise_Step_" + 8 + ".csv")));
		sc = new Scanner(new BufferedReader(
				new FileReader("/u/puchhamf/misc/jars/chemical/MAPK/Pstar/data/MCData" +  "_Step_" + 9 + ".csv")));
//				new FileReader("/u/puchhamf/misc/jars/chemical/PKA/PKAr/data/MCData" + "LessNoise" + "_Step_" + 10 + ".csv")));

		for (int i = 0; i < rows; i++) {
			String[] line = sc.nextLine().trim().split(",");
			response[i] = Double.parseDouble(line[cols]);
			for (int j = 0; j < cols; j++) {
				vars[i][j] = Double.parseDouble(line[j]);
			}
		}
		sc.close();

		// extend vars

		for (int i = 0; i < rows; i++) {
			index = 0;
			// first order:
			for (int j = 0; j < cols; j++) {
				varsExtended[i][index] = vars[i][j];
				index++;
			}

//			 second order:
			for (int j = 0; j < cols; j++) {
				for (int k = j; k < cols; k++) {
					varsExtended[i][index] = vars[i][j] * vars[i][k];
					index++;
				}
			}

////			// third order:
//			for (int j = 0; j < cols; j++) {
//				for (int k = j; k < cols; k++) {
//					for (int l = k; l < cols; l++) {
//						varsExtended[i][index] = vars[i][j] * vars[i][k] * vars[i][l];
//						index++;
//					}
//				}
//			}

		} // end i --> chains

		// REGRESSION:
		reg = LeastSquares.calcCoefficients(varsExtended, response);
		// System.out.println(reg.length);
		// Estimated Fit
		double[] fit = new double[rows];
		Arrays.fill(fit, 0.0);

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < colsExtended; j++)
				fit[i] += reg[j] * varsExtended[i][j];
		} // end i --> chains

		double rSqu = coefficientOfDetermination(response, fit);
		System.out.println("R^2 =\t" + rSqu);

		// permute

		int[] permutation = new int[rows];
		sc = new Scanner(new BufferedReader(new FileReader(perm)));
		for (int i = 0; i < rows; i++)
			permutation[i] = Integer.parseInt(sc.nextLine());
		sc.close();

		double[] fitPermuted = new double[rows];
		double rSquPermuted;
		String str = " index\t\t R^2\t\t Importance\t\tVars";
		System.out.println(str);
		boolean printAll = true;

		for (int permuteIndex = 0; permuteIndex < colsExtended; permuteIndex++) {
			str = "" + permuteIndex;
			double[][] varsPermuted = varsExtended.clone();
			for (int i = 0; i < rows; i++)
				varsPermuted[i][permuteIndex] = varsExtended[permutation[i]][permuteIndex];
			double[] regPermuted = LeastSquares.calcCoefficients(varsPermuted, response);
			// fit permutation

			Arrays.fill(fitPermuted, 0.0);
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < colsExtended; j++)
					fitPermuted[i] += regPermuted[j] * varsPermuted[i][j];

			rSquPermuted = coefficientOfDetermination(response, fitPermuted);
			str += "\t" + rSquPermuted;

			double importance = 1.0 - Math.abs(correlation(fitPermuted, fit));
			str += "\t" + importance;

//			str += "\t" + printTuples(colsExtended);
			if ((rSqu) * 0.5 > rSquPermuted || importance > 0.5 || printAll)
				System.out.println(str);
		}
		// } // end s --> steps

	}

}
