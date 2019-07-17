/*
 * Class:        OrdinaryLatticeSearch
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

import umontreal.ssj.latnetbuilder.Search;
import umontreal.ssj.latnetbuilder.weights.GeometricWeights;
import umontreal.ssj.latnetbuilder.weights.OrderDependentWeights;
import umontreal.ssj.latnetbuilder.weights.PODWeights;
import umontreal.ssj.latnetbuilder.weights.ProductWeights;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.util.Num;
import umontreal.ssj.functionfit.LeastSquares;
import umontreal.ssj.hups.Rank1Lattice;

import java.util.ArrayList;

/**
 * Class for the search of good rank-1 ordinary lattice rules using LatNet
 * Builder.
 */
public class OrdinaryLatticeSearch extends Search {

	/**
	 * {@inheritDoc}
	 */
	@Override
	public Rank1Lattice search() throws RuntimeException {
		ArrayList<String> res = executeCommandLine();
		int numPoints = Integer.parseInt(res.get(1).split("  //")[0]);
		int dimension = Integer.parseInt(res.get(2).split("  //")[0]);
		int[] genVec = new int[dimension];
		for (int coord = 0; coord < dimension; ++coord) {
			genVec[coord] = Integer.parseInt(res.get(5 + coord).split("  //")[0]);
		}
		this.merit = Double.parseDouble(res.get(5 + dimension).split("  //")[0]);
		this.time = Double.parseDouble(res.get(6 + dimension).split("  //")[0]);
		this.successful = true;
		return new Rank1Lattice(numPoints, genVec, dimension);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String pointSetType() {
		return "lattice";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int interlacing() {
		return 1;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String construction() {
		return "ordinary";
	}

	public static void main(String[] args) {
		int dim = 5;
		double a = 1.0;
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
//			weights.add(k, w*w );
//		}
			
			w = Math.pow(Math.PI * 0.5 , 0.5*(dim - 1.0)) * Math.pow(a,-dim+2.0) * Math.pow(Num.erf(a/Math.sqrt(2.0)),dim-1.0) *( Math.sqrt(0.5*Math.PI) * Num.erf(a/Math.sqrt(2.0)) - a * Math.exp(-0.5 * a * a) );
			for(int k = 2; k <= dim; ++k) {
				w *= 1.0/Math.sqrt(Math.PI * 0.5) * a * a / Num.erf(a/Math.sqrt(2.0)) *  ( Math.sqrt(0.5*Math.PI) * Num.erf(a/Math.sqrt(2.0)) - a * Math.exp(-0.5 * a * a) );
//				weights.add(k,w);
				weights.add(k,Math.sqrt(w));
			}
		
//		w = 1.0;
//		for(int k = 2; k <= dim; ++k) {
//			w = Math.pow(Math.PI * Num.erf(a*0.5)*Num.erf(a*0.5)/a,dim - k);
//			w*=Math.pow( 0.5*a * ( Math.sqrt(2.0*Math.PI) * Num.erf(a/Math.sqrt(2.0)) - 2 * a* Math.exp(-0.5 *a *a) ) ,k);
////			weights.add(k,Math.sqrt(w));
//			weights.add(k,w);
//		}

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
		
//		ProductWeights weights = new ProductWeights();
//		weights.setDefaultWeight(1.0);
		
		ArrayList<String> wList = new ArrayList<String>();
		wList.add(weights.toLatNetBuilder());

		OrdinaryLatticeSearch search = new OrdinaryLatticeSearch();
		search.setPathToLatNetBuilder("/u/puchhamf/misc/latnetbuilder/latsoft/bin/latnetbuilder");
		search.setDimension(dim);
		search.setMultilevel(false);
		search.setExplorationMethod("full-CBC");
//		search.setExplorationMethod("fast-CBC");
//		search.setFigureOfMerit("CU:P6");
		search.setFigureOfMerit("spectral");
		search.setNormType("inf");
//		search.setNormType("2");

		search.setWeights(wList);

		for (String n : N) {
			search.setPathToOutputFolder(
					"/u/puchhamf/misc/latnetbuilder/output/GenzGaussianPeak/d5a1u0p5/ord/sobFlo/rank1/spectral/" + n);
			search.setSizeParameter("" + n);
			System.out.println(search.toString());
			search.executeCommandLine();
		}

		System.out.println("A - - - O K ! ! !");
	}
}
