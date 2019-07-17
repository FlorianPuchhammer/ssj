package umontreal.ssj.latnetbuilder;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import umontreal.ssj.hups.DigitalNetBase2;
import umontreal.ssj.util.PrintfFormat;


	/**
	 * Class for the construction od digital nets.
	 */
	public class DigitalNetBase2FromLatNetBuilder extends DigitalNetBase2 {
		
		int interlacing;
		String construction;

		public DigitalNetBase2FromLatNetBuilder(int numRows, int numCols, int dim, int[] matrices) {
			this.numCols = numCols;
			this.numRows = Math.min(numRows, MAXBITS);
			this.numPoints = 1 << this.numCols;
			this.dim = dim;
			this.genMat = matrices;
			this.outDigits = MAXBITS;
			// this.outDigits = this.numRows;
			this.normFactor = 1.0 / ((double) (1L << this.outDigits));
		}
		
		public DigitalNetBase2FromLatNetBuilder(String filename, int r1, int w, int s1) throws IOException {
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

			outDigits = w;
			numCols = Integer.parseInt(res.get(0).split("  //")[0]);
			numPoints = 1 << numCols;
//			numRows = Math.min(Integer.parseInt(res.get(1).split("  //")[0]),r1);
			numRows = Integer.parseInt(res.get(1).split("  //")[0]);
			interlacing = Integer.parseInt(res.get(4).split("  //")[0]);
//			int interlacing = Integer.parseInt(res.get(4).split("  //")[0]);
			int dimension = Integer.parseInt(res.get(3).split("  //")[0]);
			construction = res.get(5).split("  //")[0];
			dim = s1;

//			System.out.println("TEST:\noffset = " + offsetForParsingGeneratingMatrix(dimension) + "\nconstruction = " + construction);
			int[][][] mats = new int[dim][numRows][numCols];
			for (int coord = 0; coord < dim; ++coord) {
				for (int row = 0; row < numRows; ++row) {
					String[] tmp = res.get(coord * (numRows + 1) + row + offsetForParsingGeneratingMatrix(dimension))
							.split(" ");
					for (int col = 0; col < numCols; ++col) {
//						System.out.println("TEST: (" + coord + ", " + row + ", " + col + ")" + "\tdim = " + dim + "\tnumRows = " + numRows + "\tnumCols = " + numCols);
						mats[coord][row][col] = Integer.parseInt(tmp[col]);
//						mats[coord][row][col] = 0;

					}
				}
			}
			dim /= interlacing;
			genMat = new int[dim * numCols];
			int trueNumRows = Math.min(31, numRows * interlacing);
			for (int coord = 0; coord < dim; ++coord) {
				for (int col = 0; col < numCols; ++col) {
					genMat[coord * numCols + col] = 0;
					for (int row = 0; row < trueNumRows; ++row) {
						// genMat[coord * numCols + col] += (1 << (trueNumRows - 1 - row)) *
						// mats[coord*interlacing + row % interlacing][row/interlacing][col];
						genMat[coord * numCols + col] += (1 << (31 - 1 - row))
								* mats[coord * interlacing + row % interlacing][row / interlacing][col];
					}
				}
			}
			this.normFactor = 1.0 / ((double) (1L << this.outDigits));
//			System.out.println("TESTinside:\n" + toString());
			

		}


		public DigitalNetBase2FromLatNetBuilder(String filename, int s1) throws IOException {
			 new DigitalNetBase2FromLatNetBuilder( filename, MAXBITS, 31, s1); 
		}
		
		public String toStringDetailed() {
		      StringBuffer sb = new StringBuffer (toString() + PrintfFormat.NEWLINE);
		      sb.append ("dim = " + dim + PrintfFormat.NEWLINE);
		      for (int i = 0; i < dim; i++) {
		         sb.append (PrintfFormat.NEWLINE + "// dim = " + (1 + i) +
		              PrintfFormat.NEWLINE);
		         for (int c = 0; c < numCols; c++)
		            sb.append  (genMat[i*numCols + c]  + PrintfFormat.NEWLINE);
		      }
		      sb.append ("--------------------------------" + PrintfFormat.NEWLINE);
		      return sb.toString ();
		   }
		
		/**
		 * Offset for the parsing of generating matrices.
		 */
		private int offsetForParsingGeneratingMatrix(int dimension) {
			if (construction.equals("Sobol")) {
//				System.out.println("TEST: Right branch!");
				return 7 + dimension;
			} else if (construction.equals("Explicit")) {
				return 7;
			} else {
//				System.out.println("TEST: Wrong branch!");
				return 8 + dimension;
			}
		}
	}

