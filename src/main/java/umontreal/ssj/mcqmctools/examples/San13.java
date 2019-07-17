package umontreal.ssj.mcqmctools.examples;
import java.io.*;
import java.util.Scanner;
import umontreal.ssj.rng.*;
import umontreal.ssj.probdist.*;
import umontreal.ssj.stat.*;
import umontreal.ssj.mcqmctools.*;


/**
 * This class simulates a specific stochastic activity network with 9 nodes and
 * 13 links, taken from Elmaghraby (1977) and used again in L'Ecuyer and Lemieux
 * (2000), "Variance Reduction via Lattice Rules". The goal is to estimate the
 * longest path.
 * This program is very specific to this example and uses a very naive way to
 * compute the shortest path, by enumerating all six paths!
 */

public class San13 implements MonteCarloModelDouble {  // implements MCModelDensity {

//	double x;
	double[] V = new double[13];
	ContinuousDistribution[] dist = new ContinuousDistribution[13];
	// We consider the 6 paths that can lead to the sink.
	double[] paths = new double[6];
	double maxPath; // Length of the current longest path.

	// The constructor reads link length distributions in a file.
	public San13( String fileName) throws IOException {
//		this.x = x;
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

	// Returns the dimension (number s of required uniforms) for this model.
	public int getDimension() {
		return 13;
	};
	
	public void deterministicT() {
		int pp = 0;
		for (int k = 0; k < 13; k++) {
			V[k] = dist[k].getMean();
			if (V[k] < 0.0)
				V[k] = 0.0;
		}
		// Path lengths
		paths[0] = V[1] + V[5] + V[10];
		paths[1] = V[0] + V[2] + V[5] + V[10];
		paths[2] = V[0] + V[4] + V[10];
		paths[3] = V[0] + V[3] + V[7] + V[9] + V[10];
		paths[4] = V[0] + V[3] + V[7] + V[8] + V[12];
		paths[5] = V[0] + V[3] + V[6] + V[11] + V[12];
		maxPath = paths[0];
		for (int p = 1; p < 6; p++) {
			System.out.println("Path number  " + p + ",   " + paths[p]);
			if (paths[p] > maxPath) {
				pp = p;
				maxPath = paths[p];
			}
		}
		System.out.println("Path number  " + pp + ",   " + maxPath);
	}

	public void simulate(RandomStream stream) {
		for (int k = 0; k < 13; k++) {
			V[k] = dist[k].inverseF(stream.nextDouble());
			if (V[k] < 0.0)
				V[k] = 0.0;
		}
		// Path lengths
		paths[0] = V[1] + V[5] + V[10];
		paths[1] = V[0] + V[2] + V[5] + V[10];
		paths[2] = V[0] + V[4] + V[10];
		paths[3] = V[0] + V[3] + V[7] + V[9] + V[10];
		paths[4] = V[0] + V[3] + V[7] + V[8] + V[12];
		paths[5] = V[0] + V[3] + V[6] + V[11] + V[12];
		maxPath = paths[0];
		for (int p = 1; p < 6; p++)
			if (paths[p] > maxPath)
				maxPath = paths[p];
	}

	public double getPerformance() {
		return maxPath;
//		if (maxPath > x)
//			return 1.0;
//		else
//			return 0.0;
	}

	public double getMin() {
//		return 0.0;
		return 30.0;
	}

	public double getMax() {
//		return 200.0;
		return 100.0;
	};


	public String toString() {
		String s = "SAN network with 9 nodes and 13 links, from Elmaghraby (1977)\n"
				+ "Return length T of longest path.\n";
		    //    + "Return SQUARE length of longest path, T * T.\n";
		    //  + "Estimate prob longest path > x = " + x + "\n";
		return s;
	}

/*
 * 	public static void main(String[] args) throws IOException {
		int n = 1000;
		San13 san = new San13(90.0, "san13a.dat");
		TallyStore statT = new TallyStore("SAN13 example");
		System.out.println("Deterministic approximation: T = "
				+ san.deterministicT() + "\n\n");
//		MCExperimentDensity.simulateRunsDefaultReport(san, n, new LFSR113(),
//				statT);
	}
 */
}
