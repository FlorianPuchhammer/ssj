package umontreal.ssj.latnetbuilder;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import umontreal.ssj.hups.Rank1Lattice;

public class Rank1LatticeFromLatNetBuilder extends Rank1Lattice {
	int offset = 5;
	
	
	
	public Rank1LatticeFromLatNetBuilder(String filename,int s) throws IOException {
		super(1,new int[s],s);
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException e) {
			System.out.println("Problem reading file: " + e);
		}

		ArrayList<String> res = new ArrayList<String>();

		String line;
		while ((line = br.readLine()) != null) {
			res.add(line);
		}
		
		this.dim = s;
		v = new double[s];
		int n = Integer.parseInt(res.get(1).split("  //")[0]);
//	      genAs = new int[s];
	      for (int j = 0; j < s; j++) {
	         genAs[j] = Integer.parseInt(res.get(offset+j));
	      }
	      initN (n);
		
	}
	

	
	private void initN (int n) {
	      numPoints = n;
	      normFactor = 1.0 / (double) n;
	      for (int j = 0; j < dim; j++) {
	         int amod = (genAs[j] % n) + (genAs[j] < 0 ? n : 0);
	         v[j] = normFactor * amod;
	      }
	   }
}
