package umontreal.ssj.stat.density;


public abstract class ConditionalDensityEstimator extends DensityEstimatorDoubleArray {

	@Override
	public void setData(double[][] data) {
		this.data = new double[data.length][];
		for(int i = 0; i < data.length; i++) {
			this.data[i] = new double[data[i].length];
			for(int j = 0; j < data[i].length; j++)
				this.data[i][j] = data[i][j];
		}
		
	}
	
	@Override
	public double evalDensity(double x) {
		double dens = 0.0;
		int N = data.length;
		double Ninv = 1.0 / (double) N;
		for (int i = 0; i < N; i++) {
			dens += evalEstimator(x, data[i]);
			dens *= Ninv;
		}
		return dens;
	}
	
	@Override
	public double[] evalDensity(double[] x) {
		int k = x.length;
		double[] dens = new double[k];
		int N = data.length;
		double Ninv = 1.0 / (double) N;
		for (int j = 0; j < k; j++) {
			dens[j] = 0.0;
			for (int i = 0; i < N; i++) {
				dens[j] += evalEstimator(x[j], data[i]);
				
			}
			dens[j] *= Ninv;
//			System.out.println("TEST:\t" + j + "\t" + evalEstimator(x[j], data[j]));

		}
		return dens;
	}
	
	
	public double[] evalDensity(double[] evalPoints, double[][] data) {
		setData(data);
		return evalDensity(evalPoints);
	}
	
	public abstract  double evalEstimator(double x, double[] data);

	@Override
	public String toString() {
		return "Conditional Density Estimator";
	}

	
}
