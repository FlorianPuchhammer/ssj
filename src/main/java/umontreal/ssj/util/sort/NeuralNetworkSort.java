/*
 * Class:        HilbertCurveSort
 * Description:  Sorts d-dimensional points in [0,1)^d based on Hilbert curve.
 * Environment:  Java
 * Software:     SSJ 
 * Copyright (C) 2014  Pierre L'Ecuyer and Universite de Montreal
 * Organization: DIRO, Universite de Montreal
 * @author       
 * @since

 * SSJ is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (GPL) as published by the
 * Free Software Foundation, either version 3 of the License, or
 * any later version.

 * SSJ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * A copy of the GNU General Public License is available at
   <a href="http://www.gnu.org/licenses">GPL licence site</a>.
 */

/* IMPORTANT NOTE:
* Much of this code has been taken (with adaptations) from  
*     the hilbert.c  code  
* Author: Spencer W. Thomas
* EECS Dept.
* University of Michigan
* Date: Thu Feb  7 1991
* Copyright (c) 1991, University of Michigan
*/
package umontreal.ssj.util.sort;

import java.util.Comparator;

import umontreal.ssj.util.sort.HilbertCurveSort.LongIndexComparator2;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

/**
 * This class implements a @ref MultiDimSort01<T extends MultiDim01> that can
 * sort an array of points in the @f$d@f$-dimensional unit
 * hypercube @f$[0,1)^d@f$, by following a Hilbert curve, and using (at most)
 * the first @f$m@f$ bits of each point. See @cite iHAM07a&thinsp;. The objects
 * sorted by this class can only be points in @f$[0,1)^d@f$, represented as
 * arrays of `double`. This sort does not apply directly to more general @ref
 * MultiDimComparable<T> objects. For that, see the class \ref
 * HilbertCurveBatchSort instead. However, this sort can be applied to points in
 * another space if we first define a mapping between this space and the unit
 * hypercube. For example, to sort points in the real space, it suffices to map
 * each coordinate to @f$[0,1)@f$.
 *
 * This sort (conceptually) divides the unit hypercube @f$[0,1)^d@f$ in
 * 
 * @f$2^{dm}@f$ subcubes of equal sizes, by dividing each axis in @f$2^m@f$
 *              equal parts, and uses the first @f$m@f$ bits of each of
 *              the @f$d@f$ coordinates to place each point in one of the
 *              subcubes. It then enumerates the subcubes in the same order as a
 *              Hilbert curve in @f$[0,1)^d@f$ would visit them, and orders the
 *              points accordingly. Each cube has an (integer) Hilbert
 *              index @f$r@f$ from 0 to @f$2^{dm}-1@f$ and the cubes (and
 *              points) are ordered according to this index. Two points that
 *              fall in the same subcube can be placed in an unspecified
 *              (arbitrary) order.
 *
 *              It may happen that some of the subcubes contain more than one
 *              point at the end. To get a sense of the probability that this
 *              happens, in the case where there are @f$n@f$ points and these
 *              points are independent and uniformly distributed
 *              over @f$[0,1)^d@f$, it is known that the number
 * @f$C@f$ of collisions (the number of times that a point falls in a box
 *         already occupied by another point when the points are generated one
 *         after the other) has approximately a Poisson distribution with mean
 * @f$\lambda_c = n^2/(2k)@f$, where @f$k = 2^{md}@f$. See
 * @cite rLEC02c&thinsp;, for example. By taking @f$m@f$ such that @f$2^{md}
 *       \geq n^2@f$, the expected number of collisions is less than 1/2 and
 *       then one can often neglect them. Otherwise, one should
 *       increase @f$m@f$. Note however that the assumption of uniformity and
 *       independence does not always hold in practice.
 *
 *       For the implementation of sorts based on the Hilbert curve (or Hilbert
 *       index), we identify and sort the subcubes by their Hilbert index, but
 *       it is also convenient to identify them (alternatively) with @f$m@f$-bit
 *       integer coordinates: The subcube with coordinates @f$(i_1,…,i_d)@f$ is
 *       defined as @f$\prod_{j=0}^{d-1} [i_j 2^{-m}, (i_j+1) 2^{-m})@f$. Note
 *       that each interval is open on the right. That is, if we multiply the
 *       coordinates of a point in the subcube by @f$2^m@f$ and truncate them to
 *       integers, we obtain the integer coordinates of the subcube. For
 *       example, if @f$d=2@f$ and @f$m=4@f$, we have @f$2^8 = 256@f$ subcubes,
 *       whose integer coordinates go from 0 to 15, and the point @f$(0.1,
 *       0.51)@f$ belongs to the subcube with integer coordinates @f$(1, 8)@f$.
 *
 *       For given @f$d@f$ and @f$m@f$, this class offers methods to compute the
 *       integer coordinates of the corresponding subcube from the real-valued
 *       coordinates of a point in @f$[0,1)^d@f$, as well as the Hilbert index
 *       of a subcube from its integer coordinates, and vice versa. The code
 *       that computes the latter correspondences is taken (with slight
 *       adaptations) from the `hilbert.c` program of Spencer W. Thomas,
 *       University of Michigan, 1991.
 *
 *       To sort a set of @f$n@f$ points in @f$[0,1)^d@f$, we first compute the
 *       integer coordinates and then the Hilbert index of the subcube for each
 *       point, then sort the points by order of Hilbert index. For the latter,
 *       we construct an index of type `long[n][2]` whose first coordinate is
 *       the point number and the second is its Hilbert index. The method
 *       #sortIndexOfLong2 is provided to sort such an index by its second
 *       coordinate. Points having the same Hilbert index are ordered
 *       arbitrarily. After this sort, the first coordinate at position @f$i@f$
 *       in the index is the (original) number of the point that comes in
 *       position @f$i@f$ when the points are sorted by Hilbert index. This
 *       index is used to reorder the original points. It can be accessed (after
 *       each sort) via the method #getIndexAfterSort(). This access can be
 *       convenient for example in case we sort a `double[][]` array with the
 *       Hilbert sort and want to apply the corresponding permutation afterward
 *       to another array of objects. Certain subclasses of HilbertCurveSort use
 *       this.
 *
 *       <div class="SSJ-bigskip"></div>
 */

public class NeuralNetworkSort implements MultiDimSortN<MultiDim> {
	static double[] w;
	int dimension; // Dimension d of the points used for the sort.
	double[] PerformanceForSort;
	NeuralNetworkMap NNMap;
	double[][] indexForSort;
	// public static int batchSize = 100;

	/*
	 * public static String fileTrain; public static String fileTest; public static
	 * int numInputs; public static int numOutputs; public static int numHiddenNodes
	 * ; public static int seed; public static double learningRate; public static
	 * int nEpochs;
	 */
	public String fileTrain;
	public String fileTest;
	public int numInputs;
	public int numOutputs;
	public int numHiddenNodes;
	public int seed;
	public double learningRate;
	public int nEpochs;
	int batchSize;

	/**
	 * Constructs a HilbertCurveSort object that will use the first
	 * 
	 * @f$m@f$ bits of each of the first `d` coordinates to sort the points. The
	 *         constructor will initialize a @ref NeuralNetworkMap with arguments
	 *         `d` and @f$m@f$. This map can be accessed with #getNeuralNetworkMap.
	 * @param d
	 *            maximum dimension
	 * @param m
	 *            number of bits used for each coordinate
	 */
	public NeuralNetworkSort(int d) {
		dimension = d;
		// NNMap = new NeuralNetworkMap(fileTrain,fileTest, numInputs, numOutputs,
		// numHiddenNodes ,seed, learningRate, nEpochs)

	}

	public NeuralNetworkSort(int d, String fileTrain, String fileTest, int numInputs, int numOutputs,
			int numHiddenNodes, int seed, double learningRate, int nEpochs, int batchSize) {
		dimension = d;
		this.batchSize = batchSize;
		this.NNMap = new NeuralNetworkMap(fileTrain, fileTest, numInputs, numOutputs, numHiddenNodes, seed,
				learningRate, nEpochs, batchSize);

	}

	/*
	 * public void sort (double[] a, int iMin, int iMax) {
	 * 
	 * indexForSort = new Integer [iMax]; for (int i=0; i< iMax;i++) indexForSort
	 * [i] = i; Arrays.sort(indexForSort, new Comparator<Integer>() {
	 * 
	 * @Override public int compare(final Integer o1, final Integer o2) { return
	 * Double.compare(a[o1], a[o2]); } }); double[] aclone = a.clone(); // Save copy
	 * of a before the sort. for (int i= iMin; i< iMax; ++i) { a[i] = aclone[(int)
	 * indexForSort[i]]; } }
	 */

	/**
	 * Sorts the subarray `a[iMin..iMax-1]` with this Hilbert curve sort. The type
	 * `T` must actually be MultiDimComparable01. This is verified in the method.
	 * 
	 * @throws InterruptedException
	 * @throws IOException
	 */

	/*
	 * public void sort (MultiDim[] a, int iMin, int iMax) { try { NNMap
	 * .trainingTesting(batchSize); } catch (IOException | InterruptedException e) {
	 * e.printStackTrace(); } PerformanceForSort = new double[iMax]; for (int i =
	 * iMin; i < iMax; ++i){ //PerformanceForSort [i] =a[i].getPerformance(); try {
	 * PerformanceForSort [i] =NNMap.prediction(a[i].getState()); } catch
	 * (FileNotFoundException e) { e.printStackTrace(); } } sort
	 * (PerformanceForSort, iMin, iMax); // Now use indexForSort to sort a. // We do
	 * not want to clone all the objects in a, // but only the array of pointers.
	 * MultiDim[] aclone = a.clone(); // new Object[iMax]; for (int i = iMin; i <
	 * iMax; ++i) a[i] = aclone[(int) indexForSort[i]]; }
	 */
	public void sort(MultiDim[] a, int iMin, int iMax) {
		/*
		 * try { NNMap .trainingTesting(batchSize); } catch (IOException |
		 * InterruptedException e) { e.printStackTrace(); } PerformanceForSort = new
		 * double[iMax]; for (int i = iMin; i < iMax; ++i){ //PerformanceForSort [i]
		 * =a[i].getPerformance(); try { PerformanceForSort [i]
		 * =NNMap.prediction(a[i].getState()); } catch (FileNotFoundException e) {
		 * e.printStackTrace(); } }
		 */
		double b[][] = new double[iMax][dimension];
		for (int i = iMin; i < iMax; ++i)
			b[i] = a[i].getState();
		System.out.println("Hello0");
		sort(b, iMin, iMax);
		// Now use indexForSort to sort a.
		// We do not want to clone all the objects in a,
		// but only the array of pointers.
		MultiDim[] aclone = a.clone(); // new Object[iMax];
		for (int i = iMin; i < iMax; ++i)
			a[i] = aclone[(int) indexForSort[i][0]];
	}

	/**
	 * Sorts the entire array: same as `sort (a, 0, a.length)`.
	 */

	public void sort(MultiDim[] a) {
		sort(a, 0, a.length);
	}

	/**
	 * Returns the index computed by the last sort, which is sorted by the second
	 * coordinate. It contains (in the first coordinates of its entries) the
	 * permutation made by that sort.
	 */
	public double[][] getIndexAfterSort() {
		return indexForSort;
	}

	/**
	 * Returns the dimension of the unit hypercube.
	 */
	public int dimension() {
		return dimension;
	}

	public String toString() {
		return "NeuralNetwork";
	}

	public double LinearCombination(double[] a) {
		double sum = 0.0;
		for (int i = 0; i < a.length; i++)
			sum += a[i] * w[i];
		return sum;
	}

	/*
	 * public void sort (double[][] a, int iMin, int iMax) { if (iMin+1 == iMax)
	 * return; indexForSort = new Integer[iMax]; // Index used for sort. double b[]
	 * = new double[iMax];
	 * 
	 * for (int i=0; i < a.length; ++i) { // Transform to integer coordinates. b[i]
	 * = LinearCombination(a[i]); } // Sort the index based on the positions on the
	 * Hilbert curve sort (b, iMin, iMax); // Now use the sorted index to sort a.
	 * double[][] aclone = a.clone(); // Save copy of a before the sort. for (int i=
	 * iMin; i< iMax; ++i) { a[i] = aclone[(int) indexForSort[i]]; } }
	 */
	public void sort(double[][] a, int iMin, int iMax) { //a --> chains
		if (iMin + 1 == iMax)
			return;
		indexForSort = new double[iMax][2];

		try {
			NNMap.trainingTesting(batchSize);
			System.out.println("Hello1");
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}

		for (int i = 0; i < a.length; ++i) {

			try {
				indexForSort[i][0] = i;
				indexForSort[i][1] = NNMap.prediction(a[i]); // Hilbert index of this point.
				System.out.println("Sort " + indexForSort[i][1]);
				// System.out.println("Hello2");
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}

		sortIndexOfDouble2(indexForSort, iMin, iMax);
		// Now use the sorted index to sort a.
		double[][] aclone = a.clone(); // Save copy of a before the sort.
		for (int i = iMin; i < iMax; ++i) {
			a[i] = aclone[(int) indexForSort[i][0]];
		}
	}

	public static class DoubleIndexComparator2 implements Comparator<double[]> {

		public int compare(double[] p1, double[] p2) {
			if (p1[1] > p2[1])
				return 1;
			else if (p1[1] < p2[1])
				return -1;
			else
				return 0;
		}
	}

	/**
	 * Sorts the `index` table by its second coordinate.
	 */
	public static void sortIndexOfDouble2(double[][] index, int iMin, int iMax) {
		// if (iMin==(iMax-1)) return;
		Arrays.sort(index, iMin, iMax, new DoubleIndexComparator2());
	}

	public void sort(double[][] a) {
		sort(a, 0, a.length);
	}

}
