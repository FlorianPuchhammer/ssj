/*
 * Class:        DigitalNetSearch
 * Description:
 * Environment:  Java
 * Software:     SSJ
 * Copyright (C) 2018  Pierre L'Ecuyer and Universite de Montreal
 * Organization: DIRO, Universite de Montreal
 * @author Maxime Godin and Pierre Marion
 * @since August 2018
 *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package umontreal.ssj.latnetbuilder;

import umontreal.ssj.functionfit.LeastSquares;
import umontreal.ssj.hups.DigitalNetBase2;
import umontreal.ssj.hups.DigitalNetFromFile;
import umontreal.ssj.latnetbuilder.weights.GeometricWeights;
import umontreal.ssj.latnetbuilder.weights.OrderDependentWeights;
import umontreal.ssj.latnetbuilder.weights.PODWeights;
import umontreal.ssj.latnetbuilder.weights.ProductWeights;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.util.Num;
import umontreal.ssj.util.PrintfFormat;

import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.StreamTokenizer;
import java.lang.Math;

/**
 * Class for the search of good digital nets using LatNet Builder.
 */
public class DigitalNetSearch extends Search {

	/**
	 * Class for the construction od digital nets.
	 */
	public class DigitalNetBase2FromLatNetBuilder extends DigitalNetBase2 {

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
			dimension = Integer.parseInt(res.get(3).split("  //")[0]);
			dim = s1;

//			System.out.println("TEST:\ndim = " + dim + "\tnumRows = " + numRows + "\tnumCols = " + numCols);
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
			dim = dim / interlacing;
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

			System.out.println("TESTinside:\n" + toStringDetailed());

		}

		public DigitalNetBase2FromLatNetBuilder(String filename, int s1) throws IOException {
			this(filename, MAXBITS, 31, s1);
		}

		public String toStringDetailed() {
			StringBuffer sb = new StringBuffer(toString() + PrintfFormat.NEWLINE);
			sb.append("dim = " + dim + PrintfFormat.NEWLINE);
			for (int i = 0; i < dim; i++) {
				sb.append(PrintfFormat.NEWLINE + "// dim = " + (1 + i) + PrintfFormat.NEWLINE);
				for (int c = 0; c < numCols; c++)
					sb.append(genMat[i * numCols + c] + PrintfFormat.NEWLINE);
			}
			sb.append("--------------------------------" + PrintfFormat.NEWLINE);
			return sb.toString();
		}
	}

	String construction;
	int interlacing;

	/**
	 * Constructor.
	 * 
	 * @param construction Type of construction (eg. sobol, explicit, polynomial,
	 *                     ...).
	 */
	public DigitalNetSearch(String construction) {
		super();
		this.construction = construction;
		this.interlacing = 1;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public DigitalNetBase2 search() throws RuntimeException {
		ArrayList<String> res = executeCommandLine();
		int numCols = Integer.parseInt(res.get(0).split("  //")[0]);
		int numRows = Integer.parseInt(res.get(1).split("  //")[0]);
		int interlacing = Integer.parseInt(res.get(4).split("  //")[0]);
		int dimension = Integer.parseInt(res.get(3).split("  //")[0]);
		int[][][] mats = new int[dimension][numRows][numCols];
		for (int coord = 0; coord < dimension; ++coord) {
			for (int row = 0; row < numRows; ++row) {
				String[] tmp = res.get(coord * (numRows + 1) + row + offsetForParsingGeneratingMatrix(dimension))
						.split(" ");
				for (int col = 0; col < numCols; ++col) {
					mats[coord][row][col] = Integer.parseInt(tmp[col]);
				}
			}
		}

		dimension = dimension / interlacing;

		int[] genMat = new int[dimension * numCols];
		int trueNumRows = Math.min(31, numRows * interlacing);
		for (int coord = 0; coord < dimension; ++coord) {
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
		this.merit = Double.parseDouble(res.get(res.size() - 2).split("  //")[0]);
		this.time = Double.parseDouble(res.get(res.size() - 1).split("  //")[0]);
		this.successful = true;
		return new DigitalNetBase2FromLatNetBuilder(trueNumRows, numCols, dimension, genMat);
	}

	/**
	 * Offset for the parsing of generating matrices.
	 */
	private int offsetForParsingGeneratingMatrix(int dimension) {
		if (construction == "sobol") {
			return 7 + dimension;
		} else if (construction == "explicit") {
			return 7;
		} else {
			return 8 + dimension;
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String pointSetType() {
		return "net";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int interlacing() {
		return interlacing;
	}

	/**
	 * Sets the interlacing factor of the searched digital net.
	 * 
	 * @param interlacing Interlacing factor.
	 */
	public void setInterlacing(int interlacing) {
		this.interlacing = interlacing;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String construction() {
		return construction;
	}

	/**
	 * Sets the construciton method of the searched digital net.
	 * 
	 * @param construction Type of construction (eg. sobol, explicit, polynomial,
	 *                     ...).
	 */
	public void setConstruction(String construction) {
		this.construction = construction;
	}

	public static void main(String[] args) {
		int dim = 5;
		double a = 0.5;
		String[] N = { "2^13", "2^14", "2^15", "2^16", "2^17", "2^18", "2^19", "2^20" };

		OrderDependentWeights weights = new OrderDependentWeights();
		weights.add(1, 0);
		double w;
//		for (int k = 2; k <= dim; ++k) {
//			//GFunction
//			w = Math.pow(3.0 * (a + 1.0) * (a + 1.0), -k);
			// GenzGaussian
//			double rho = 0.5;
//			w = Math.pow(Math.PI / (a * a), (double) dim) * Math.pow(rho, k);
//			w *= Math.pow(2.0 * NormalDist.cdf01(a / Math.sqrt(2.0)) - 1.0, 2.0 * (dim - k));
//			w *= Math.pow(a / Math.sqrt(2.0 * Math.PI) * (2.0 * NormalDist.cdf01(a) - 1.0)
//					- Math.pow(2.0 * NormalDist.cdf01(a / Math.sqrt(2.0)) - 1.0, 2.0), k);
//			weights.add(k, w*w);
//		}
			
//			w = Math.pow(Math.PI * 0.5 , 0.5*(dim - 1.0)) * Math.pow(a,-dim+2.0) * Math.pow(Num.erf(a/Math.sqrt(2.0)),dim-1.0) *( Math.sqrt(0.5*Math.PI) * Num.erf(a/Math.sqrt(2.0)) - a * Math.exp(-0.5 * a * a) );
//			for(int k = 2; k <= dim; ++k) {
//				w *= 1.0/Math.sqrt(Math.PI * 0.5) * a * a / Num.erf(a/Math.sqrt(2.0)) *  ( Math.sqrt(0.5*Math.PI) * Num.erf(a/Math.sqrt(2.0)) - a * Math.exp(-0.5 * a * a) );
//				weights.add(k,w);
//			}
		
		w = 1.0;
		for(int k = 2; k <= dim; ++k) {
			w *= 0.05;
			weights.add(k,w);
		}

//		double[] x = new double[dim-1];
//		double[] y = new double[dim-1];
//		double[] leastSquares = new double[dim-1];
//		for(int j = 2; j < dim+1; ++j) {
//			x[j-2] = j-2;
//			y[j-2] = -j * Num.log2(3.0 * (a+1.0) * (a+1.0));
//		}
//		
//		leastSquares = LeastSquares.calcCoefficients(x,y);
//
//		double c = Math.pow(2.0,leastSquares[0]);
//		double rho = Math.pow(2.0,leastSquares[1]);
//		GeometricWeights weights = new GeometricWeights(rho*rho,dim,c*c);
////		GeometricWeights weights = new GeometricWeights(rho,dim,c);

		weights.setDefaultWeight(0.0);
		ArrayList<String> wList = new ArrayList<String>();
		wList.add(weights.toLatNetBuilder());

		DigitalNetSearch search = new DigitalNetSearch("sobol");
		search.setPathToLatNetBuilder("/u/puchhamf/misc/latnetbuilder/latsoft/bin/latnetbuilder");
		search.setDimension(dim);
		search.setMultilevel(false);
//		search.setExplorationMethod("full-CBC");
		search.setExplorationMethod("exhaustive");
		search.setExplorationMethod("random:500");
		search.setFigureOfMerit("t-value");
//		search.setFigureOfMerit("CU:P6");
		search.setNormType("inf");
//		search.setNormType("2");
//		search.setInterlacing(4); search.setFigureOfMerit("CU:IA4"); 

		search.setWeights(wList);

		for (String n : N) {
			search.setPathToOutputFolder(
					"/u/puchhamf/misc/latnetbuilder/output/GenzGaussianPeak/d5a2u0p5/ord/0p05k/sob/t500/" + n);
			search.setSizeParameter("" + n);
			System.out.println(search.toString());
			search.executeCommandLine();
		}

		System.out.println("A - - - O K ! ! !");
	}

}
