package flo.biologyArrayRQMC.examples;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import flo.neuralNet.NeuralNet;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.probdist.PoissonDist;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.util.Chrono;

import umontreal.ssj.util.sort.MultiDim01;

public class SchloeglSystemProjected extends ChemicalReactionNetwork implements MultiDim01 {

	double N0; // Total number of molecules
	double[][] means = { // mean obtained from MC-files (for hilbert sort)
			{0.0,0.0},
			{250.388, 99819.6},
			{250.876, 99638.8},
			{251.564, 99457.4},
			{252.604, 99275.1},
			{253.987, 99091.4},
			{255.527, 98906.4},
			{256.891, 98720.2},
			{257.754, 98533.2},
			{257.838, 98346.3},
			{257.059, 98160.2},
			{255.55, 97975.6},
			{253.3, 97793.2},
			{250.431, 97613.6},
			{247.077, 97437.2}
	}
	; 
	double [][] stdDevs = {
			{1.0,1.0},
			{24.5894, 17.8271},
			{38.9162, 46.7408},
			{53.3392, 89.9386},
			{68.5439, 147.717},
			{84.3616, 220.439},
			{100.014, 307.709},
			{114.611, 408.377},
			{127.464, 520.577},
			{138.278, 642.256},
			{146.982, 771.466},
			{153.793, 906.387},
			{158.854, 1045.46},
			{162.331, 1187.25},
			{164.538, 1330.62}
	}; // std devs for hilbert sort

	public SchloeglSystemProjected(double[] c, double[] X0, double tau, double T, double N0) {
		this.c = c;
		this.X0 = X0;
		this.tau = tau;
		this.T = T;
		S = new double[][] { { 1, -1, 1, -1 }, { -1, 1, 0, 0 }, { 0, 0, -1, 1 } };
		init();
		this.N0 = N0;
	}

	@Override
	public int compareTo(MarkovChainComparable m, int i) {
		if (!(m instanceof SchloeglSystemProjected)) {
			throw new IllegalArgumentException("Can't compare an SchloeglSystem with other types of Markov chains.");
		}
		double mx;

		mx = ((SchloeglSystemProjected) m).X[i];
		return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));
	}

	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append(" SchloeglSystem:\n");
		sb.append("X0 =\t" + "{" + X0[0] + ", " + X0[1] + ", " + (N0 - X0[0] - X0[1]) + "}\n");
		sb.append("c =\t" + "{" + c[0] + ", " + c[1] + ", " + c[2] + "}\n");
		sb.append("T =\t" + T + "\n");
		sb.append("tau =\t" + tau + "\n");
		sb.append("steps =\t" + numSteps + "\n");
		sb.append("----------------------------------------------\n\n");

		return sb.toString();
	}

	@Override
	public double getPerformance() {
		return X[0];
//			return X[1];
	}

	public void genDataPoly(String dataLabel, int n, int numSteps, RandomStream stream) throws IOException {
		double[][][] states = new double[n][][];
		double[] performance = new double[n];
		simulRunsWithSubstreams(n, numSteps, stream, states, performance);
		StringBuffer sb;
		FileWriter fw;
		File file;
		for (int step = 0; step < numSteps; step++) {
			sb = new StringBuffer("{");
			file = new File(dataLabel + "_Step_" + step + "poly.txt");
			file.getParentFile().mkdirs();
			fw = new FileWriter(file);

			for (int i = 0; i < n; i++) {
				sb.append("{");
				for (int j = 0; j < getStateDimension(); j++)
					sb.append(states[i][step][j] + ",");

				sb.append(performance[i] + "},\n");
			}
			sb.deleteCharAt(sb.lastIndexOf(","));
			sb.append("}");
			fw.write(sb.toString());
			fw.close();
			System.out.println("*******************************************");
			System.out.println(" STEP " + step);
			System.out.println("*******************************************");
			System.out.println(sb.toString());
		}
	}

	@Override
	public double[] getPoint() {
		double[] state01 = new double[N];
		for (int i = 0; i < N; i++)
			state01[i] = getCoordinate(i);
		return state01;
	}

	@Override
	public void computePropensities() {
		double x2 = (N0 - X[0] - X[1]);
		a[0] = 0.5 * c[0] * X[0] * (X[0] - 1.0) * X[1];
		a[1] = c[1] * X[0] * (X[0] - 1.0) * (X[0] - 2.0) / 6.0;
		a[2] = c[2] * x2;
		a[3] = c[3] * X[0];
	}

	@Override
	public double getCoordinate(int j) {
		
		double zvalue;
		double x02 = N0 - X0[0] - X0[1];
//		switch (j) {
//		case 0:
////			zvalue = (X[j] - X0[j]
////					- (step - 1.0) * tau * tau
////							* (0.5 * c[0] * X0[0] * (X0[0] - 1.0) * X0[1]
////									- c[1] * X0[0] * (X0[0] - 1.0) * (X0[0] - 2.0) / 6.0 + c[2] * x02 - c[3] * X0[0]))
////					/ (tau * Math.sqrt((step - 1.0) * (0.5 * c[0] * X0[0] * (X0[0] - 1.0) * X0[1]
////							+ c[1] * X0[0] * (X0[0] - 1.0) * (X0[0] - 2.0) / 6.0 + c[2] * x02 + c[3] * X0[0])));
//			return NormalDist.cdf01(zvalue);
//		case 1:
//			zvalue = (X[j] - X0[j]
//					- (step - 1.0) * tau * tau
//							* (-0.5 * c[0] * X0[0] * (X0[0] - 1.0) * X0[1]
//									+ c[1] * X0[0] * (X0[0] - 1.0) * (X0[0] - 2.0) / 6.0))
//					/ (tau * Math.sqrt((step - 1.0) * (0.5 * c[0] * X0[0] * (X0[0] - 1.0) * X0[1]
//							+ c[1] * X0[0] * (X0[0] - 1.0) * (X0[0] - 2.0) / 6.0)));
//			return NormalDist.cdf01(zvalue);
//		default:
//			throw new IllegalArgumentException("Invalid state index");
//		}
		return NormalDist.cdf01( (X[j] - means[step][j]) / stdDevs[step][j] );
		}

	public static void main(String[] args) throws IOException {
		ChemicalReactionNetwork model;

		double[] c = { 3E-7, 1E-4, 1E-3, 3.5 };
		double[] x0 = { 250.0, 1E5 };
		double N0 = 2E5 + 1E5 + 250.0;
		double T = 4.0;
		double tau = 0.2;

		model = new SchloeglSystemProjected(c, x0, tau, T, N0);
		;
		String dataFolder = "data/SchloeglSystemProj/";
		model.init();

		NeuralNet test = new NeuralNet(model, dataFolder); // This is the array of comparable chains.

		System.out.println(model.toString());

		int numChains = 524288 * 2;
		// int numChains = 100;
		int logNumChains = 19 + 1;

		Chrono timer = new Chrono();
		RandomStream stream = new MRG32k3a();

		/*
		 ***********************************************************************
		 ************* BUILD DATA***********************************************
		 ***********************************************************************
		 */
		// boolean genData = true;

		// String dataLabel = "SobData";
		String dataLabel = "MCData";

		// PointSet sobol = new SobolSequence(logNumChains, 31, model.numSteps *
		// model.getK());
		// PointSetRandomization rand = new LMScrambleShift(stream);
		// RQMCPointSet p = new RQMCPointSet(sobol, rand);

		// if (genData) {
		timer.init();
		// test.genData(dataLabel, numChains, model.numSteps, p.iterator());
		test.genData(dataLabel, numChains, model.numSteps, stream);
		System.out.println("\n\nTiming:\t" + timer.format());
		// }
	}

}
