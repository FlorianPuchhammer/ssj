/*
 * Class:        PolynomialLatticeSearch
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

import java.util.ArrayList;

import umontreal.ssj.functionfit.LeastSquares;
import umontreal.ssj.latnetbuilder.DigitalNetSearch;
import umontreal.ssj.latnetbuilder.weights.GeometricWeights;
import umontreal.ssj.latnetbuilder.weights.OrderDependentWeights;
import umontreal.ssj.latnetbuilder.weights.PODWeights;
import umontreal.ssj.latnetbuilder.weights.ProductWeights;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.util.Num;

/**
 * Class for the search of good polynomial lattice rules using LatNet Builder.
 */
public class PolynomialLatticeSearch extends DigitalNetSearch {

	String pointSetType;

	/**
	 * Constructor.
	 * 
	 * @param pointSetType Point set type (lattice or net). Used to switch between
	 *                     the two implementation of polynomial lattice rules in
	 *                     LatNet Builder.
	 */
	public PolynomialLatticeSearch(String pointSetType) {
		super("polynomial");
		this.pointSetType = pointSetType;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String pointSetType() {
		return pointSetType;
	}

	/**
	 * Changes the point set type to use when searching.
	 * 
	 * @param pointSetType Point set type (lattice or net).
	 */
	public void changePointSetTypeView(String pointSetType) {
		this.pointSetType = pointSetType;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void setConstruction(String construction) {
		return;
	}

	public static void main(String[] args) {
		int dim = 5;
		double a = 0.5;
		String[] N = { "2^13", "2^14", "2^15", "2^16", "2^17", "2^18", "2^19", "2^20" };

		OrderDependentWeights weights = new OrderDependentWeights();
		weights.add(1, 0);
		double w;
//		for (int k = 2; k <= dim; ++k) {
////			//GFunction
////			w = Math.pow(3.0 * (a + 1.0) * (a + 1.0), -k);
//			// GenzGaussian
//			double rho = 0.5;
//			w = Math.pow(Math.PI / (a * a), (double) dim) * Math.pow(rho, k);
//			w *= Math.pow(2.0 * NormalDist.cdf01(a / Math.sqrt(2.0)) - 1.0, 2.0 * (dim - k));
//			w *= Math.pow(a / Math.sqrt(2.0 * Math.PI) * (2.0 * NormalDist.cdf01(a) - 1.0)
//					- Math.pow(2.0 * NormalDist.cdf01(a / Math.sqrt(2.0)) - 1.0, 2.0), k);
//			weights.add(k, w * w);
//		}
		
//		w = Math.pow(Math.PI * 0.5 , 0.5*(dim - 1.0)) * Math.pow(a,-dim+2.0) * Math.pow(Num.erf(a/Math.sqrt(2.0)),dim-1.0) *( Math.sqrt(0.5*Math.PI) * Num.erf(a/Math.sqrt(2.0)) - a * Math.exp(-0.5 * a * a) );
//		for(int k = 2; k <= dim; ++k) {
//			w *= 1.0/Math.sqrt(Math.PI * 0.5) * a * a / Num.erf(a/Math.sqrt(2.0)) *  ( Math.sqrt(0.5*Math.PI) * Num.erf(a/Math.sqrt(2.0)) - a * Math.exp(-0.5 * a * a) );
//			weights.add(k,w);
//		}

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

		ArrayList<String> wList = new ArrayList<String>();
		wList.add(weights.toLatNetBuilder());

		PolynomialLatticeSearch search = new PolynomialLatticeSearch("lattice");
		search.setPathToLatNetBuilder("/u/puchhamf/misc/latnetbuilder/latsoft/bin/latnetbuilder");
		search.setDimension(dim);
		search.setMultilevel(false);
		search.setExplorationMethod("fast-CBC");
		search.setFigureOfMerit("CU:R");
		search.setNormType("2");
//		search.setPathToOutputFolder("/u/puchhamf/misc/latnetbuilder/output/poly/interlaced4/");
//		search.setInterlacing(4); search.setFigureOfMerit("CU:IA4"); 

		search.setWeights(wList);

		for (String n : N) {
			search.setPathToOutputFolder(
					"/u/puchhamf/misc/latnetbuilder/output/GenzGaussianPeak/d5a2u0p5/ord/0p05k/poly/R/" + n);
			search.setSizeParameter("" + n);
			System.out.println(search.toString());
			search.executeCommandLine();
		}

		System.out.println("A - - - O K ! ! !");

	}
}
