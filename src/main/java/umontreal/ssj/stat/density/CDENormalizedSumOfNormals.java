package umontreal.ssj.stat.density;

import java.util.Arrays;

import umontreal.ssj.probdist.NormalDist;

public class CDENormalizedSumOfNormals extends ConditionalDensityEstimator {

	/*
	 * Array containing d weights, where the d is the number of estimators.
	 */
	private double[] weights;
	private int dimension;
	private double sigma;
	private double[] sigmas;
	

	public CDENormalizedSumOfNormals(int dim, double[] weights,double []sigmas) {
		this.weights = new double[dim];
//		this.setWeights(weights);
		this.weights = weights;
		this.setDimension(dim);
		this.sigmas = sigmas;
		this.sigma = 0.0;
		for(double s: sigmas)
			sigma +=s*s;
		sigma = Math.sqrt(sigma);
	}
	
	public CDENormalizedSumOfNormals(int dim, double[] weights) {
		this.setWeights(weights);
		this.setDimension(dim);
	}

	public CDENormalizedSumOfNormals(int dim) {
		weights = new double[dim];
		Arrays.fill(weights, 1.0 / (double) dim);
		this.setDimension(dim);

	}

	public CDENormalizedSumOfNormals(double[] weights) {
		this(weights.length, weights);
	}

	/**
	 * Setter for the weights. 
	 * 
	 * @param weights
	 */
	public void setWeights(double[] weights) {
		for (int i = 0; i < dimension; i++)
			this.weights[i] = weights[i];
//		normalizeWeights();
	}

	private void normalizeWeights() {
		double sum = 0.0;
		for (double w : weights)
			sum += w;
		for (int i = 0; i < dimension; i++)
			weights[i] /= sum;
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
//		for(double d : weights)
//		System.out.println("w = " + d);
		double val = 0.0;
		double sqrtDim = Math.sqrt((double) dimension);
		double sum;
		for (int leave = 0; leave < getDimension(); ++leave) {
			sum = 0.0;
			if (weights[leave] > 0) {

				for (int j = 0; j < getDimension(); ++j)
					if (j != leave)
						sum += data[j];
//				val += weights[leave] * NormalDist.density01(x * sqrtDim - sum) * sqrtDim;
				val += weights[leave] * NormalDist.density01(x * sigma/ sigmas[leave] - sum/sigmas[leave]) * sigma/sigmas[leave];
			} // endif
		}
		return val;
	}

	public String toString() {
		return "CDENormalizedSumOf" + getDimension() + "Normals";
	}

}
