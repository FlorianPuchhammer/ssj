
package umontreal.ssj.util.sort;

import java.util.Comparator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;

import umontreal.ssj.util.sort.HilbertCurveSort.LongIndexComparator2;
import umontreal.ssj.util.sort.NeuralNetworkSort.DoubleIndexComparator2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

/**
 * Class that implements a sorting algorithm for possibly higher dimensional
 * objects based on neural networks. The sorting algorithm uses a neural network
 * as a sort of <tt>importance function</tt> to assign a one-dimensional value
 * to the object w.r.t. which the objects are sorted.
 * 
 * @author puchhamf
 *
 */

public class NeuralNetworkSortFlo implements MultiDimSort<MultiDim> {

	/**
	 * The NN that acts as importance function.
	 */
	private MultiLayerNetwork network;
	/**
	 * The normalizer associated to the net (i.e., the normalizer with which the training data had been normalized).
	 */
	private DataNormalization normalizer;
	/**
	 * The dimension of the object to be sorted.
	 */
	private int dimension;
	/**
	 * Just a utility for sorting.
	 * TODO: get rid of this and make sorting more efficient!
	 */
	private double[][] indexForSort;

	/**
	 * Constructor that loads a NN and its associated normalizer from the file \a fileNameWithPath.
	 * @param fileNameWithPath the filename including the path.
	 * @param dim the dimension of the object to be sorted.
	 * @throws IOException
	 */
	public NeuralNetworkSortFlo(String fileNameWithPath, int dim) throws IOException {
		File file = new File(fileNameWithPath);
		network = ModelSerializer.restoreMultiLayerNetwork(file);
		normalizer = ModelSerializer.restoreNormalizerFromFile(file);
		dimension = dim;

	}
/**
 * Constructs an instance of this class for a given NN \a network with associated normalizer \a normalizer to sort
 * objects of dimension \a dim.
 * @param network the NN used for sorting.
 * @param normalizer the normalizer associated to the NN.
 * @param dim the dimension of the objects to be sorted.
 */
	public NeuralNetworkSortFlo(MultiLayerNetwork network, DataNormalization normalizer, int dim) {
		this.network = network;
		this.normalizer = normalizer;
		dimension = dim;
	}

	/**
	 * Evaluates the network at the array \a state.
	 * @param state the input.
	 * @return the network evaluated at \a state.
	 */
	public double evalNetwork(double[] state) {

		INDArray input = Nd4j.create(state);
		normalizer.transform(input);
		INDArray output = network.output(input, false);
		// normalizer.revertLabels(output);
		return output.getDouble(0);

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
		return "NeuralNetworkSort";
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
		Arrays.sort(index, iMin, iMax, new DoubleIndexComparator2());
	}

	@Override
	public void sort(double[][] a, int iMin, int iMax) {
		if (iMin + 1 == iMax)
			return;
		indexForSort = new double[iMax][2];

		for (int i = iMin; i < iMax; ++i) {
			indexForSort[i][0] = i;
			indexForSort[i][1] = evalNetwork(a[i]);
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
