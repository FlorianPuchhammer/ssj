package flo.biologyArrayRQMC.examples;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import flo.neuralNet.NeuralNet;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.util.Chrono;

public class SchloeglSystem extends ChemicalReactionNetwork {

	public SchloeglSystem(double[] c, double[] X0, double tau, double T) {
		this.c = c;
		this.X0 = X0;
		this.tau = tau;
		this.T = T;
		S = new double[][] { { 1, -1, 1, -1 }, { -1, 1, 0, 0 }, { 0, 0, -1, 1 } };
		init();
	}

	@Override
	public int compareTo(MarkovChainComparable m, int i) {
		if (!(m instanceof SchloeglSystem)) {
			throw new IllegalArgumentException("Can't compare an SchloeglSystem with other types of Markov chains.");
		}
		double mx;

		mx = ((SchloeglSystem) m).X[i];
		return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));
	}

	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append(" SchloeglSystem:\n");
		sb.append("X0 =\t" + "{" + X0[0] + ", " + X0[1] + ", " + X0[2] + "}\n");
		sb.append("c =\t" + "{" + c[0] + ", " + c[1] + ", " + c[2] + "}\n");
		sb.append("T =\t" + T + "\n");
		sb.append("tau =\t" + tau + "\n");
		sb.append("steps =\t" + numSteps + "\n");
		sb.append("----------------------------------------------\n\n");

		return sb.toString();
	}

	@Override
	public void computePropensities() {
		a[0] = 0.5 * c[0] * X[0] * (X[0] - 1.0) * X[1];
		a[1] = c[1] * X[0] * (X[0] - 1.0) * (X[0] - 2.0) / 6.0;
		a[2] = c[2] * X[2];
		a[3] = c[3] * X[0];
	}

	@Override
	public double getPerformance() {
		return X[0];
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

	public static void main(String[] args) throws IOException, InterruptedException {
		ChemicalReactionNetwork model;

		double[] c = { 3E-7, 1E-4, 1E-3, 3.5 };
		double[] x0 = { 250.0, 1E5, 2E5 };
		double T = 4.2;
		double tau = 0.2;

		model = new SchloeglSystem(c, x0, tau, T);
		;
		String dataFolder = "data/SchloeglSystem/";
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
