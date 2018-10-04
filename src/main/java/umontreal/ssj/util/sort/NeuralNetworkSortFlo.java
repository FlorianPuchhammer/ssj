
package umontreal.ssj.util.sort;

import java.util.Comparator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;

import umontreal.ssj.util.sort.HilbertCurveSort.LongIndexComparator2;
import umontreal.ssj.util.sort.NeuralNetworkSort.DoubleIndexComparator2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;



public class NeuralNetworkSortFlo implements MultiDimSortN<MultiDim> {
//	static double[] w;
//	int dimension; // Dimension d of the points used for the sort.
//	double[] PerformanceForSort;
//	NeuralNetworkMap NNMap;
//	double[][] indexForSort;

//	public String fileTrain;
//	public String fileTest;
//	public int numInputs;
//	public int numOutputs;
//	public int numHiddenNodes;
//	public int seed;
//	public double learningRate;
//	public int nEpochs;
//	int batchSize;

	private MultiLayerNetwork network; 
	private DataNormalization normalizer;
	private int dimension;
	
	public NeuralNetworkSortFlo(String fileNameWithPath, int dim) throws IOException {
		File file = new File(fileNameWithPath);
		network = ModelSerializer.restoreMultiLayerNetwork(file);
		normalizer = ModelSerializer.restoreNormalizerFromFile(file);
		dimension = dim;

	}

	public NeuralNetworkSortFlo(MultiLayerNetwork network,	DataNormalization normalizer,int dim) {
		this.network = network;
		this.normalizer = normalizer;
		dimension = dim;
	}


	public double evalNetwork(double [] state) {
		
		INDArray input = Nd4j.create(state);
		normalizer.transform(input);
		 return network.output(input).getDouble(0);

		
	}
	
	public void sort(MultiDim[] a, int iMin, int iMax) {
		if(iMin == iMax)
			return;
		double b[][] = new double[iMax][dimension];
		for (int i = iMin; i < iMax; ++i) {
			b[i] = a[i].getState();
		}
		sort(b,iMin,iMax);
		
		// Now use indexForSort to sort a.
		// We do not want to clone all the objects in a,
		// but only the array of pointers.
		MultiDim[] aclone = a.clone(); // new Object[iMax];
		for (int i = iMin; i < iMax; ++i)
			a[i] = aclone[(int) b[i][0]];
	}

	/**
	 * Sorts the entire array: same as `sort (a, 0, a.length)`.
	 */

	public void sort(MultiDim[] a) {
		sort(a, 0, a.length);
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

	@Override
	public void sort(double[][] a, int iMin, int iMax) {
		if(iMin+1 == iMax)
			return;
		double b[][] = new double[iMax][2];
		
		for (int i = iMin; i < iMax; ++i) {
			b[i][0] = i;
			b[i][1] = evalNetwork(a[i]);
		}
		Arrays.sort(b, iMin, iMax, new DoubleIndexComparator2());
		
		// Now use indexForSort to sort a.
		// We do not want to clone all the objects in a,
		// but only the array of pointers.
		double[][] aclone = a.clone(); // new Object[iMax];
		for (int i = iMin; i < iMax; ++i)
			a[i] = aclone[(int) b[i][0]];
		
	}

	@Override
	public void sort(double[][] a) {
		sort(a,0,a.length);
		
	}

//	public void sort(double[][] a) {
//		sort(a, 0, a.length);
//	}

}
