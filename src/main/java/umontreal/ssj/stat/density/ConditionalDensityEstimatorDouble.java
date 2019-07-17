package umontreal.ssj.stat.density;

public abstract class ConditionalDensityEstimatorDouble extends DensityEstimator {

	public abstract double evalEstimator(double x, double data);

	@Override
	public void setData(double[] data) {
		int n = data.length;
		this.data = new double[n];
		for (int i = 0; i < n; i++)
			this.data[i] = data[i];

	}

	@Override
	public String toString() {
		return "Conditional Density Estimator";
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
				dens[j] *= Ninv;
			}

		}
		return dens;
	}

}
