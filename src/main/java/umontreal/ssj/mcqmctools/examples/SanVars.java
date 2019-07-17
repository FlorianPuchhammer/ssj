package umontreal.ssj.mcqmctools.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.probdist.ContinuousDistribution;
import umontreal.ssj.probdist.DistributionFactory;
import umontreal.ssj.rng.RandomStream;

public class SanVars implements MonteCarloModelDoubleArray {

	double[] performance;
	ContinuousDistribution[] dist;
	
	public SanVars(String fileName) throws IOException {
		performance = new double[13];
		dist = new ContinuousDistribution[13];
		readDistributions(fileName);
	}
	
	public void readDistributions(String fileName) throws IOException {
		// Reads data and construct arrays.
		BufferedReader input = new BufferedReader(new FileReader(fileName));
		Scanner scan = new Scanner(input);
		for (int k = 0; k < 13; k++) {
			dist[k] = DistributionFactory.getContinuousDistribution(scan
					.nextLine());
			// gen[k] = new RandomVariateGen (stream, dist);
		}
		scan.close();
	}
	
	@Override
	public void simulate(RandomStream stream) {
		for (int k = 0; k < 13; k++) {
			performance[k] = dist[k].inverseF(stream.nextDouble());
			if (performance[k] < 0.0)
				performance[k] = 0.0;
		}
	}

	@Override
	public double[] getPerformance() {
		return performance;
	}

	@Override
	public int getPerformanceDim() {
		return 13;
	}
	
	@Override
	public String toString() {
		return "SAN13";
	}

}
