
package flo.optionsArrayRQMC;

import java.util.Comparator;

import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.util.sort.DoubleArrayComparator;
import umontreal.ssj.util.sort.MultiDim;
import umontreal.ssj.util.sort.MultiDimComparable;
import umontreal.ssj.util.sort.MultiDimComparator;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.MultiDimSortComparable;
import umontreal.ssj.util.sort.SplitSort;

import java.util.Arrays;

public class HestonModelMixedSort implements MultiDimSort<MultiDim> {

	/**
	 * The dimension of the object to be sorted.
	 */
	protected int dimension = 3;
	double a, b;
	MultiDimSort<MarkovChainComparable> sort;

	/**
	 * Just a utility for sorting. TODO: get rid of this and make sorting more
	 * efficient!
	 */
	private double[][] indexForSort;

	HestonModelMixedSort(double a, double b) {
		this.a = a;
		this.b = b;
		sort = new SplitSort<MarkovChainComparable>(dimension - 1);
	}

	HestonModelMixedSort(double a, double b, MultiDimSort<MarkovChainComparable> sort) {
		this.a = a;
		this.b = b;
		this.sort = sort;
	}

//	private static class DoubleIndexComparator2 implements Comparator<double[]> {
//
//		public int compare(double[] p1, double[] p2) {
//			if (p1[1] > p2[1])
//				return 1;
//			else if (p1[1] < p2[1])
//				return -1;
//			else
//				return 0;
//		}
//	}

	public void sort(MultiDim[] aa, int iMin, int iMax) {
		if (iMin == iMax)
			return;
		double bb[][] = new double[iMax][dimension];
		for (int i = iMin; i < iMax; ++i) {
//			for(int j = 0; j < aa[i].getState().length; j++)
//			bb[i][j] = aa[i].getState()[j];
			bb[i] = aa[i].getState();
		}
		sort(bb, iMin, iMax);

		// Now use indexForSort to sort a.
		// We do not want to clone all the objects in a,
		// but only the array of pointers.
		MultiDim[] aclone = aa.clone(); // new Object[iMax];
//		MultiDim[] aclone = new MultiDim[aa.l]; // new Object[iMax];

		for (int i = iMin; i < iMax; ++i)
			aa[i] = aclone[(int) indexForSort[i][2]];
	}

	public double scoreFunction(double[] bb) {
		return a * bb[0] + b * bb[1];
	}

	/**
	 * Sorts the entire array: same as `sort (a, 0, a.length)`.
	 */

	public void sort(MultiDim[] aa) {
		sort(aa, 0, aa.length);
	}

	/**
	 * Returns the dimension of the objects to be sorted.
	 */
	public int dimension() {
		return dimension;
	}

	public String toString() {
		return "MultDimToOneDimSort";
	}

	/**
	 * Sorts the `index` table by its second coordinate.
	 */
//	public static void sortIndexOfDouble2(double[][] index, int iMin, int iMax) {
//		Arrays.sort(index, iMin, iMax, new DoubleIndexComparator2());
//	}

	@Override
	public void sort(double[][] a, int iMin, int iMax) {
		if (iMin + 1 == iMax)
			return;
		indexForSort = new double[iMax][dimension];

		for (int i = iMin; i < iMax; ++i) {
			indexForSort[i][0] = scoreFunction(a[i]);
			indexForSort[i][1] = a[i][2];
			indexForSort[i][2] = i;
		}
		sort.sort(indexForSort, iMin, iMax);
//		Arrays.sort(indexForSort, iMin, iMax, new DoubleIndexComparator2());

		// Now use indexForSort to sort a.
		// We do not want to clone all the objects in a,
		// but only the array of pointers.
		double[][] aclone = a.clone(); // new Object[iMax];
//		double[][] aclone = new double[a.length][a[0].length]; // new Object[iMax];
//		for (int i = iMin; i < iMax; ++i)
//			for (int j = 0; j < aclone[i].length; j++)
//				aclone[i][j] = a[i][j];

		for (int i = iMin; i < iMax; ++i) {
			a[i] = aclone[(int) indexForSort[i][2]];
		}
	}

	@Override
	public void sort(double[][] a) {
		sort(a, 0, a.length);

	}

}
