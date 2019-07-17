package umontreal.ssj.mcqmctools.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.probdist.ContinuousDistribution;
import umontreal.ssj.probdist.DistributionFactory;
import umontreal.ssj.rng.RandomStream;

public class SanVarsCDE implements MonteCarloModelDoubleArray {

	double[] V = new double[13];
	ContinuousDistribution[] dist = new ContinuousDistribution[13];
	// We consider the 6 paths that can lead to the sink.
	double[] paths = new double[5];
	double maxPath; // Length of the current longest path.

	// The constructor reads link length distributions in a file.
	public SanVarsCDE(String fileName) throws IOException {

		readDistributions(fileName);
	}

	private void readDistributions(String fileName) throws IOException {
		// Reads data and construct arrays.
		BufferedReader input = new BufferedReader(new FileReader(fileName));
		Scanner scan = new Scanner(input);
		for (int k = 0; k < 13; k++) {
			dist[k] = DistributionFactory.getContinuousDistribution(scan.nextLine());
		}
		scan.close();
	}

	public void simulate(RandomStream stream) {
		for (int k = 0; k < 13; k++) {
			V[k] = dist[k].inverseF(stream.nextDouble());
			if (V[k] < 0.0)
				V[k] = 0.0;
		}

		paths[0] = V[0] + V[10]; // corresponds to path over Y4

		double temp = V[0] + V[2]; // corresponds to path over Y5
		if (temp > V[1])
			paths[1] = temp;
		else
			paths[1] = V[1];
		paths[1] += V[10];

		paths[2] = V[0] + V[3] + V[11] + V[12]; // corresponds to path over Y6

		paths[3] = V[0] + V[3] + V[7] + V[12]; // corresponds to path over Y8

		paths[4] = V[0] + V[3] + V[7] + V[10];// corresponds to path over Y9

	}

	public double[] getPerformance() {
		return paths;
	}

	public int getDimension() {
		return 13;
	}

	
	public String toString(){
		return "SAN13";
	}




	@Override
	public int getPerformanceDim() {
		return 5;
	}

}
