package umontreal.ssj.stat.density;

import java.util.ArrayList;

import umontreal.ssj.probdist.ContinuousDistribution;
import umontreal.ssj.stat.PgfDataTable;

/**
 * Same as @ref DensityEstimator but here the data points are \f$t\f$-dimensional.
 * @author florian
 *
 */
public abstract class DensityEstimatorDoubleArray {

	/**
	 * The data associated with this DensityEstimatorDoubleArray object, if any.
	 */
	protected double[][] data;
	
	/**
	 * Sets the observations for the density estimator to \a data. Note that, in
	 * some cases, this requires to completely reconstruct the density estimator.
	 * 
	 * @param data
	 *            the desired observations.
	 */
	public abstract void setData(double[][] data);
	
	/**
	 * Gives the observations for this density estimator, if any.
	 * 
	 * @return the observations for this density estimator.
	 */
	public double[][] getData() {
		return data;
	}
	
	/**
	 * Evaluates the density estimator at \a x.
	 * 
	 * @param x
	 *            the evaluation point.
	 *
	 * @return the density estimator evaluated at \f$x\f$.
	 */

	public abstract double evalDensity(double x);
	
	/**
	 * Evaluates the density estimator at the points in \a evalPoints. By default,
	 * this method calls `evalDensity(double)` for each entry of \a evalPoints. Many
	 * density estimators can handle evaluation at a set of points more efficiently
	 * than that. If so, it is suggested to override this method in the
	 * implementation of the corresponding estimator.
	 *
	 * @return the density estimator evaluated at the points \a evalPoints.
	 */
	public double[] evalDensity(double[] evalPoints) {
		int k = evalPoints.length;
		double[] dens = new double[k];
		for (int j = 0; j < k; j++)
			dens[j] = evalDensity(evalPoints[j]);
		return dens;
	}
	
	/**
	 * Sets the observations for the density estimator to \a data and evaluates the
	 * density at each point in \a evalPoints.
	 * 
	 * @param evalPoints
	 *            the evaluation points.
	 * @param data
	 *            the observations.
	 * @return the density estimator defined by \a data evaluated at each point in
	 *         \a evalPoints.
	 */
	public double[] evalDensity(double[] evalPoints, double[][] data) {
		setData(data);
		return evalDensity(evalPoints);
	}
	
	/**
	 * This method is particularly designed to evaluate the density estimator in
	 * such a way that the result can be easily used to estimate the empirical IV
	 * and other convergence-related quantities.
	 * 
	 * Assume that we have \f$m\f$ independent realizations of the underlying model.
	 * For each such realization this method constructs a density and evaluates it
	 * at the points from \a evalPoints. The independent realizations are passed via
	 * the 3-dimensional \f$m\times n \times t\f$ array \a data, where \f$n\f$ denotes the
	 * number of observations per realization and \f$t\f$ the model dimension. Hence, 
	 * its first index identifies the
	 * independent realization while its second index identifies a specific
	 * observation of this realization.
	 * 
	 * The result is returned as a \f$m\times k\f$ matrix, where \f$k \f$ is the
	 * number of evaluation points, i.e., the length of \a evalPoints. The first
	 * index, again, identifies the independent realization whereas the second index
	 * corresponds to the point of \a evalPoints at which the density estimator was
	 * evaluated.
	 * 
	 *
	 * @param evalPoints
	 *            the evaluation points.
	 * @param data
	 *            the three-dimensional array carrying the observations of \f$m\f$
	 *            independent realizations of the underlying model.
	 *
	 * @return the density estimator for each realization evaluated at \a
	 *         evalPoints.
	 */
	public double[][] evalDensity(double[] evalPoints, double[][][] data) {
		int m = data.length;
		double[][] density = new double[m][];
		for (int r = 0; r < m; r++)
			density[r] = evalDensity(evalPoints, data[r]);

		return density;
	}
	
	/**
	 * This function is particularly designed for experiments with many different
	 * types of density estimators, as it evaluates all of these estimators at the
	 * points in \a evalPoints. To this end, the user passes a list of density
	 * estimators in \a listDE as well as \f$m\f$ independent realizations of the
	 * underlying model consisting of \f$n\f$ observations each in the \f$m\times
	 * n \times t\f$ array \a data.
	 * 
	 * This method then calls `evalDensity(double[], double[][][])` for
	 * each density estimator in \a listDE, thus evaluating the respective density
	 * estimator at the \f$k\f$ points in \a evalPoints and adds the resulting
	 * \f$m\times k\f$ array to \a listDensity.
	 * 
	 * @param listDE
	 *            the list of density estimators.
	 * @param evalPoints
	 *            the evaluation points.
	 * @param data
	 *            the three-dimensional array carrying the observations of \f$m\f$
	 *            independent realizations of the underlying model.
	 * @param listDensity
	 *            a list to which the evaluations at \a evalPoints of each density
	 *            estimator in \a listDE are added.
	 * 
	 */
	public static void evalDensity(ArrayList<DensityEstimatorDoubleArray> listDE, double[] evalPoints, double[][][] data,
			ArrayList<double[][]> listDensity) {
		for (DensityEstimatorDoubleArray de : listDE)
			listDensity.add(de.evalDensity(evalPoints, data));
	}
	
	/**
	 * This method computes the empirical variance based on the values given in \a
	 * data. More precisely, \a density is a \f$m\times k\f$ matrix, whose entries
	 * correspond to \f$m\f$ independent realizations of the density estimator, each
	 * evaluated at \f$k\f$ evaluation points. Such a matrix can, for instance, be
	 * obtained by `evalDensity(double[], double[][][])`.
	 * 
	 * The empirical variance is computed at each of those \f$k \f$ evaluation
	 * points and returned in an array of size \f$k\f$.
	 * 
	 * @param density
	 *            the estimated density of \f$m\f$ independent realizations of the
	 *            estimator, each evaluated at \f$k\f$ evaluation points.
	 * @return the empirical variance at those \f$k\f$ evaluation points.
	 */
	public static double[] computeVariance(double[][] density) {
		
		return DensityEstimator.computeVariance(density);
	}
	
	/**
	 * This method estimates the empirical IV over the interval \f$[a,b]\f$. Based
	 * on the density estimates of \f$m\f$ independent replications of the density
	 * estimator evaluated at \f$k\f$ evaluation points, which are provided by \a
	 * density, it computes the empirical variance at each evaluation point and
	 * stores it in \a variance.
	 * 
	 * To estimate the empirical IV, we sum up the variance at the evaluation points
	 * \f$x_1,x_2,\dots,x_k\f$ and multiply by \f$(b-a)/k\f$, i.e.
	 * 
	 * \f[ \int_a^b \hat{f}(x)\mathrm{d}x \approx \frac{b-a}{k} \sum_{j =
	 * 1}^k\hat{f}(x_j), \f]
	 * 
	 * 
	 * where \f$\hat{f}\f$ denotes the density estimator. In other words, we
	 * approximate the empirical IV by an equally weighted quadrature rule using the
	 * aforementioned evaluation points as integration nodes.
	 * 
	 * 
	 * Note that this is only an approximation of the true empirical IV and that the
	 * approximation quality significantly depends on the choice of evaluation
	 * points.
	 * 
	 * The data for the variance are given in the two-dimensional \f$m\times k\f$
	 * array \a density, which is also described in `computeVariance(double[][])` and
	 * can be obtained by `evalDensity(double[], double[][])` . The
	 * boundaries of the interval are given by \a a and \a b. Note that the array \a
	 * variance needs to be of length \f$k\f$.
	 * 
	 * 
	 * @param density
	 *            the \f$m\times k\f$ array that contains the data of evaluating
	 *            \f$m\f$ replicates of the density estimator at \f$k\f$ evaluation
	 *            points
	 * @param a
	 *            the left boundary of the interval.
	 * @param b
	 *            the right boundary of the interval.
	 * @param variance
	 *            the array of length \f$k\f$ in which the variance at each
	 *            evaluation point is stored.
	 * @return the estimated empirical IV over \f$[a,b]\f$.
	 */
	public static double computeIV(double[][] density, double a, double b, double[] variance) {
		variance = computeVariance(density);
		int k = density[0].length;
		double iv = 0.0;
		for (double var : variance)
			iv += var;
		return iv * (b - a) / (double) k;
	}

	/**
	 * This method estimates the empirical IV over the interval \f$[a,b]\f$ for a
	 * collection of different estimators. In \a densityList the user passes a list
	 * of \f$m\times k\f$ arrays which contain the density estimates of \f$m\f$
	 * independent replications of each density estimator evaluated at \f$k\f$
	 * evaluation points. Such a list can be obtained via `evalDensity(ArrayList,
	 * double[], double[][][], ArrayList)` , for instance.
	 * 
	 * The method then calls `computeIV(double[][], double, double, double[])` for
	 * each element of \a densityList and adds the thereby obtained estimated
	 * empirical IV to the list that is being returned.
	 * 
	 * 
	 * @param listDensity
	 *            list containing \f$m\times k\f$ arrays that contain the data of
	 *            evaluating \f$m\f$ replicates of each density estimator at \f$k\f$
	 *            evaluation points \a evalPoints.
	 * @param a
	 *            the left boundary of the interval.
	 * @param b
	 *            the right boundary of the interval.
	 * @param listIV
	 *            the list to which the estimated empirical IV of each density
	 *            estimator will be added.
	 *
	 */
	public static void computeIV(ArrayList<double[][]> listDensity, double a, double b, ArrayList<Double> listIV) {
		DensityEstimator.computeIV(listDensity, a, b, listIV);
	}
	
	/**
	 * In situations where the true density is known this method can estimate the
	 * empirical MISE over the interval \f$[a,b]\f$. This can be particularly
	 * interesting and useful for testing density estimators. Since it is necessary
	 * to compute either the ISB or the IV to get the MISE and as there is not much
	 * computational overhead to estimate the other, an array containing the
	 * estimated empirical IV, the ISB, and MISE in exactly this order is returned.
	 * Based on the density estimates of \f$m\f$ independent replications of the
	 * density estimator evaluated at \f$k\f$ evaluation points \a evalPoints, which
	 * are provided by \a density, it computes the empirical variance, the
	 * square-bias, and the mean square error (MSE) at each evaluation point and
	 * stores the result in \a variance, \a sqBias, and \a mse, respectively. It is
	 * important that the evaluation points in \a evalPoints are the same as the
	 * ones used to construct \a density.
	 * 
	 * To estimate the empirical IV and MISE we sum up the variance and the MSE at
	 * the \f$k\f$ evaluation points and multiply by \f$(b-a)/k\f$, i.e. we
	 * approximate the empirical IV by an equally weighted quadrature rule with \a
	 * evalPoints as integration nodes. The ISB is then computed as the difference
	 * of the MISE and the IV. Note that this is only an approximation of the true
	 * empirical values and that the approximation quality significantly depends on
	 * the choice of \a evalPoints.
	 * 
	 * The data for the variance and mse are given in the two-dimensional \f$m\times
	 * k\f$ array \a density, which is also described in
	 * `computeVariance(double[][])` and can be obtained by `evalDensity(double[],
	 * double[][], double, double)`, and the true density is passed via a \ref
	 * umontreal.ssj.probdist.ContinuousDistribution. The evaluation points are
	 * contained in \a evalPoints and the boundaries of the interval over which we
	 * estimate are given by \a a and \a b. Note that the arrays \a variance, \a
	 * sqBias, and \a mse all need to be of length \f$k\f$.
	 * 
	 * @param dist
	 *            the true density.
	 * @param evalPoints
	 *            the \f$k\f$ evaluation points.
	 * @param density
	 *            the \f$m\times k\f$ array that contains the data of evaluating
	 *            \f$m\f$ replicates of the density estimator at \f$k\f$ evaluation
	 *            points \a evalPoints.
	 * @param a
	 *            the left boundary of the interval.
	 * @param b
	 *            the right boundary of the interval.
	 * @param variance
	 *            the array of length \f$k\f$ in which the variance at each
	 *            evaluation point is stored.
	 * @param sqBias
	 *            the array of length \f$k\f$ in which the square-bias at each
	 *            evaluation point is stored.
	 * @param mse
	 *            the array of length \f$k\f$ in which the MSE at each evaluation
	 *            point is stored.
	 * @return an array containing the estimated empirical IV, ISB, and MISE in
	 *         exactly this order.
	 */
	public static double[] computeMISE(ContinuousDistribution dist, double[] evalPoints, double[][] density, double a,
			double b, double[] variance, double[] sqBias, double[] mse) {
		
		return DensityEstimator.computeMISE(dist, evalPoints, density, a, b, variance, sqBias, mse);
	}
	
	/**
	 * This method estimates the empirical MISE over the interval \f$[a,b]\f$ for a
	 * collection of different estimators. This can be done when the true density is
	 * actually known and is particularly interesting and/or useful for testing
	 * density estimators.
	 * 
	 * In \a densityList the user passes a list of \f$m\times k\f$ arrays which
	 * contain the density estimates of \f$m\f$ independent replications of each
	 * density estimator evaluated at \f$k\f$ evaluation points. Such a list can be
	 * obtained by calling `evalDensity(ArrayList, double[], double[][], ArrayList)`, for instance.
	 * 
	 * The method then calls `computeMISE(ContinuousDistribution, double[],
	 * double[][], double, double, double[], double[], double[])` for each element of
	 * \a listDensity. This results in an array containing the estimated empirical
	 * IV, ISB, and MISE in exactly this order, which is then added to the list \a
	 * listMISE.
	 * 
	 * @param dist
	 *            the true density.
	 * @param evalPoints
	 *            the \f$k\f$ evaluation points.
	 * @param listDensity
	 *            list of \f$m\times k\f$ arrays that contain the data of evaluating
	 *            \f$m\f$ replicates of each density estimator at \f$k\f$ evaluation
	 *            points \a evalPoints.
	 * @param a
	 *            the left boundary of the interval.
	 * @param b
	 *            the right boundary of the interval.
	 * @param listMISE
	 *            a list to which the arrays containing the estimated empirical IV,
	 *            ISB, and MISE of each density estimator are added.
	 */

	public static void computeMISE(ContinuousDistribution dist, double[] evalPoints, ArrayList<double[][]> listDensity,
			double a, double b, ArrayList<double[]> listMISE) {
		DensityEstimator.computeMISE(dist, evalPoints, listDensity, a, b, listMISE);
	}
	/**
	 * Gives a short description of the estimator.
	 * 
	 * @return a short description.
	 */
	public abstract String toString();

	/**
	 * Gives a plot of the estimated density. The \f$x\f$-values are passed in \a
	 * evalPoints and the \f$y\f$-values in \a density. The user may also set the
	 * title of the plot via \a plotTitle as well as the names of the axes via \a
	 * axisTitles. The latter contains the name of the \f$x\f$ axis as first element
	 * and the name of the \f$y\f$ axis as second.
	 * 
	 * The plot itself is returned as a string, which forms a stand-alone LaTex file
	 * (including necessary headers) implementing a tikZ picture.
	 * 
	 * This function merely tailors and simplifies the methods provided by \ref
	 * umontreal.ssj.stat.PgfDataTable for the purpose of plotting a univariate
	 * function. If the user seeks to produce more sophisticated plots, please refer
	 * to the aforementioned class.
	 * 
	 * @param evalPoints
	 *            the \f$x\f$-values.
	 * @param density
	 *            the \f$y\f$-values.
	 * @param plotTitle
	 *            the title of the plot.
	 * @param axisTitles
	 * @return
	 */
	public static String plotDensity(double[] evalPoints, double[] density, String plotTitle, String[] axisTitles) {
		double[][] plotData = new double[evalPoints.length][];
		for(int i = 0; i < evalPoints.length; i++) {
			plotData[i] = new double [2];
			plotData[i][0] =evalPoints[i];
			plotData[i][1] = density[i];
		}
		PgfDataTable table = new PgfDataTable(plotTitle, "", axisTitles, plotData);
		StringBuffer sb = new StringBuffer("");
		sb.append(PgfDataTable.pgfplotFileHeader());
		sb.append(table.drawPgfPlotSingleCurve(plotTitle, "axis", 0, 1, 2, "", ""));
		sb.append(PgfDataTable.pgfplotEndDocument());
		return sb.toString();
	}

	/**
	 * Estimates the roughness functional
	 * 
	 * \f[ R(g) = \int_a^b g^2(x)\mathrm{d}x\f]
	 * 
	 * of a function \f$g\f$ over the interval \f$[a,b]\f$. This is done via a
	 * quadrature rule using predetermined values of \f$g\f$ passed by the user via
	 * \a density as integration nodes.
	 * 
	 * @param density
	 *            the function evaluations.
	 * @param a
	 *            the left boundary of the interval
	 * @param b
	 *            the right boundary of the interval
	 * @return
	 */
	public static double roughnessFunctional(double[] density, double a, double b) {
		double fac = (b - a) / (double) density.length;
		double sum = 0.0;
		for (double d : density) {
			sum += d * d;
		}
		return fac * sum;
	}
	protected static double [] genEvalPoints(int numPts, double a, double b) {
		double[] evalPts = new double[numPts];
		double invNumPts = 1.0 / ((double) numPts);
		for(int i = 0; i <numPts; i++)
			evalPts[i] = a + (b-a) * ( (double)i + 0.5 ) * invNumPts;
		return evalPts;
	}
}
