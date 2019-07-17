package umontreal.ssj.stat.density;

import java.util.Arrays;

public class LLNormalizedSumOfNormals extends ConditionalDensityEstimator {

	/*
	 * Array containing d weights, where the d is the number of estimators.
	 */

	private int dimension;

	public LLNormalizedSumOfNormals(int dim) {
		this.setDimension(dim);
	}

	
	/**
	 * @return the dimension
	 */
	public int getDimension() {
		return dimension;
	}

	/**
	 * @param dimension the dimension to set
	 */
	public void setDimension(int dimension) {
		this.dimension = dimension;
	}

	@Override
	public double evalEstimator(double x, double[] data) {
		double sum = 0.0;
		double sumSq = 0.0;
		for (int j = 0; j < getDimension(); j++) {
			sum += data[j];
		}
		if (sum > x)
			return 0;
		else {
			for(int j = 0; j < getDimension(); j++)
				sumSq += -data[j]*data[j];
			return ( (sumSq + getDimension())/x );
		}
			
	}
	
	public String toString() {
		return "LLNormalizedSumOf" + getDimension() + "normals";
	}

}
