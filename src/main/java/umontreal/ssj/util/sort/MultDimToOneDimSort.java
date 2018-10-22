package umontreal.ssj.util.sort;

import java.util.Arrays;
import java.util.Comparator;


public abstract class MultDimToOneDimSort implements MultiDimSort<MultiDim> {
	
	/**
	 * The dimension of the object to be sorted.
	 */
	protected int dimension;
	/**
	 * Just a utility for sorting.
	 * TODO: get rid of this and make sorting more efficient!
	 */
	private double[][] indexForSort;

	private static class DoubleIndexComparator2 implements Comparator<double[]> {

		public int compare(double[] p1, double[] p2) {
			if (p1[1] > p2[1])
				return 1;
			else if (p1[1] < p2[1])
				return -1;
			else
				return 0;
		}
	}
	
	public void sort(MultiDim[] a, int iMin, int iMax) {
		if (iMin == iMax)
			return;
		double b[][] = new double[iMax][dimension];
		for (int i = iMin; i < iMax; ++i) {
			b[i] = a[i].getState();
		}
		sort(b, iMin, iMax);

		// Now use indexForSort to sort a.
		// We do not want to clone all the objects in a,
		// but only the array of pointers.
		MultiDim[] aclone = a.clone(); // new Object[iMax];
		for (int i = iMin; i < iMax; ++i)
			a[i] = aclone[(int) indexForSort[i][0]];
	}
	
	public abstract double scoreFunction(double [] b);
	
	/**
	 * Sorts the entire array: same as `sort (a, 0, a.length)`.
	 */

	public void sort(MultiDim[] a) {
		sort(a, 0, a.length);
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
	public static void sortIndexOfDouble2(double[][] index, int iMin, int iMax) {
		Arrays.sort(index, iMin, iMax, new DoubleIndexComparator2());
	}

	@Override
	public void sort(double[][] a, int iMin, int iMax) {
		if (iMin + 1 == iMax)
			return;
		indexForSort = new double[iMax][2];

		for (int i = iMin; i < iMax; ++i) {
			indexForSort[i][0] = i;
			indexForSort[i][1] = scoreFunction(a[i]);
		}
		Arrays.sort(indexForSort, iMin, iMax, new DoubleIndexComparator2());

		// Now use indexForSort to sort a.
		// We do not want to clone all the objects in a,
		// but only the array of pointers.
		double[][] aclone = a.clone(); // new Object[iMax];
		for (int i = iMin; i < iMax; ++i) {
			a[i] = aclone[(int) indexForSort[i][0]];
		}
	}

	@Override
	public void sort(double[][] a) {
		sort(a, 0, a.length);

	}

	
}