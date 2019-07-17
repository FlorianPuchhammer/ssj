package flo.biologyArrayRQMC.examples;

import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.util.sort.MultiDim01;

public class MAPK extends ChemicalReactionNetwork implements MultiDim01 {
	
	double[][] means = {
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{2307.3, 3485.34, 4790.97, 4475.55, 5275.46, 8748.14, 2322.37, 	3475.61, 4790.9, 5468.62, 7764.86},
			{2213.12, 2937.2, 4994.44, 4451.72, 4290.1, 9116.64, 2242.11, 2918.51, 4995.21, 5437.91, 8148.37},
			{2166.54, 2598.95, 5155.75, 4428.35, 3631.48, 9316.95, 2208.9, 2571.91, 5157.67, 5407.71, 8362.7},
			{2146.9, 2376.16, 5293.09, 4405.39, 3151.74, 9425.36, 2202.24, 2341.36, 5296.35, 5377.97, 8484.32},
			{2143.56, 2224.09, 5415.76, 4382.79, 2782.29, 9477.36, 2211.52, 2181.88, 5420.5, 5348.62, 8549.0},
			{2150.39, 2118.34, 5529.04, 4360.54, 2486.49, 9492.08, 2230.68, 2069.01, 5535.35, 5319.67, 8575.98},
			{2163.53, 2044.37, 5636.27, 4338.6, 2242.82, 9480.75, 2255.92, 1988.19, 5644.18, 5291.1, 8576.53},
			{2180.36, 1992.82, 5739.64, 4316.98, 2037.7, 9450.56, 2284.64, 1930.02, 5749.2, 5262.88, 8557.9},
			{2199.16, 1957.59, 5840.6, 4295.67, 1862.25, 9406.15, 2315.08, 1888.23, 5851.91, 5235.01, 8524.85}
	};
	
	double[][] stdDevs = {
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{20.3496, 36.8298, 17.7744, 6.1631, 42.4481, 31.8676, 20.748, 36.7018, 17.7907, 6.69783, 31.6347},
			{25.4342, 39.7104, 21.4825, 8.6941, 43.2894, 35.3468, 26.003, 39.4799, 21.5159, 9.43947, 34.9615},
			{28.4866, 40.0611, 23.4257, 10.624, 42.014, 36.6507, 29.2061, 39.7768, 23.4819, 11.5262, 36.1455},
			{30.7464, 39.9742, 24.7524, 12.2301, 40.6608, 37.5014, 31.5824, 39.6764, 24.8229, 13.2673, 36.8832},
			{32.5607, 39.8318, 25.7988, 13.6326, 39.4811, 38.1961, 33.4916, 39.4542, 25.9026, 14.7936, 37.4752},
			{34.0559, 39.7144, 26.7035, 14.8874, 38.4475, 38.8148, 35.1001, 39.2972, 26.8325, 16.1536, 38.0226},
			{35.3688, 39.6968, 27.5193, 16.0249, 37.5712, 39.4666, 36.4787, 39.2225, 27.685, 17.3871, 38.586},
			{36.5228, 39.7542, 28.3107, 17.0837, 36.7761, 40.0911, 37.7118, 39.2186, 28.5088, 18.5133, 39.154},
			{37.5624, 39.8325, 29.0768, 18.0622, 36.0772, 40.7281, 38.7858, 39.3017, 29.3175, 19.5637, 39.757}
	};
	
	public MAPK(double[] c, double[] X0, double tau, double T) {
		this.c = c;
		this.X0 = X0;
		this.tau = tau;
		this.T = T;
		S = new double[][] { { -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
				{ -1, 1, 0, 1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0 }, { 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, -1, 1, 0, 0, 0, 1, 0, -1, 1, 0 },
				{ 0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, -1, 1, 1 }, { 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1 } };
		init();
	}
	


	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append("MAPK cascade model:\n");
		sb.append("Number of reactions K = " + K + "\n");
		sb.append("Number of species N = " + N + "\n");
		sb.append("X0 =\t" + "{" + X0[0]);
		for (int i = 1; i < X0.length; i++)
			sb.append(", " + X0[i]);
		sb.append("}\n");

		sb.append("c =\t" + "{" + c[0]);
		for (int i = 1; i < c.length; i++)
			sb.append(", " + c[i]);
		sb.append("}\n");
		sb.append("T =\t" + T + "\n");
		sb.append("tau =\t" + tau + "\n");
		sb.append("steps =\t" + numSteps + "\n");
		sb.append("----------------------------------------------\n\n");

		return sb.toString();
	}

	@Override
	public int compareTo(MarkovChainComparable m, int i) {
		if (!(m instanceof MAPK)) {
			throw new IllegalArgumentException("Can't compare a MAPK with other types of Markov chains.");
		}
		double mx;

		mx = ((MAPK) m).X[i];
		return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));
	}

	@Override
	public void computePropensities() {
		a[0] = c[0] * X[0] * X[1];
		a[1] = c[1] * X[2];
		a[2] = c[2] * X[2];
		a[3] = c[3] * X[3];
		a[4] = c[4] * X[1] * X[4];
		a[5] = c[5] * X[5];
		a[6] = c[6] * X[5];
		a[7] = c[7] * X[6] * X[7];
		a[8] = c[8] * X[8];
		a[9] = c[9] * X[8];
		a[10] = c[10] * X[9];
		a[11] = c[11] * X[4] * X[7];
		a[12] = c[12] * X[10];
		a[13] = c[13] * X[10];

	}

	@Override
	public double[] getPoint() {
		double[] state01 = new double[N];
		for (int i = 0; i < N; i++)
			state01[i] = getCoordinate(i);
		return state01;
	}

	@Override
	public double getCoordinate(int j) {
		double zvalue;

//		switch (j) {
//		case 0:
//			zvalue = (X[j] - X0[j] - step * tau * (-c[0] * X0[0] * X0[1] + c[1] * X0[2] + c[13] * X0[10]))
//					/ (step * tau * Math.sqrt(c[0] * X0[0] * X0[1] + c[1] * X0[2] + c[13] * X0[10]));
//
//			return NormalDist.cdf01(zvalue);
//		case 1:
//			zvalue = (X[j] - X0[j]
//					- step * tau
//							* (-c[0] * X0[0] * X0[1] + c[1] * X0[2] + c[3] * X0[3] - c[4] * X0[1] * X0[4] + c[5] * X0[5]
//									+ c[6] * X0[5] + c[1] * X0[2]))
//					/ (step * tau * Math.sqrt(c[0] * X0[0] * X0[1] + c[1] * X0[2] + c[3] * X0[3] + c[4] * X0[1] * X0[4] + c[5] * X0[5]
//							+ c[6] * X0[5] + c[1] * X0[2] ));
//			return NormalDist.cdf01(zvalue);
//		case 2:
//			zvalue  = (X[j] - X0[j] - step * tau * (c[0] * X0[0] * X0[1] - c[1] * X0[2] - c[2] * X0[2] ))
//					/ (step * tau * Math.sqrt(c[0] * X0[0] * X0[1] + c[1] * X0[2] + c[2] * X0[2] ));
//			return NormalDist.cdf01(zvalue);
//		case 3:
//			zvalue  = (X[j] - X0[j] - step * tau * ( c[2] * X0[2] -c[3] * X0[3] ))
//			/ (step * tau * Math.sqrt(c[2] * X0[2] +c[3] * X0[3] ));
//			return NormalDist.cdf01(zvalue);
//		case 4:
//			zvalue  = (X[j] - X0[j] - step * tau * ( c[2] * X0[2] -c[4] * X0[1]*X0[4]+ c[5] * X0[5] + c[9]*X0[8] - c[11]*X0[4] * X0[7] + c[12] * X0[10]))
//			/ (step * tau * Math.sqrt(c[2] * X0[2] +c[4] * X0[1]*X0[4]+ c[5] * X0[5] + c[9]*X0[8] + c[11]*X0[4] * X0[7] + c[12] * X0[10]));
//			return NormalDist.cdf01(zvalue);
//		case 5:
//			zvalue  = (X[j] - X0[j] - step * tau * ( c[4] * X0[1] * X0[4] -c[5] * X0[5] -c[6]*X0[5] ))
//			/ (step * tau * Math.sqrt( c[4] * X0[1] * X0[4] +c[5] * X0[5] +c[6]*X0[5] ));
//			return NormalDist.cdf01(zvalue);
//		case 6:
//			zvalue  = (X[j] - X0[j] - step * tau * ( c[6]  * X0[4] -c[7] * X0[6]*X0[7] +c[8]*X0[8] ))
//			/ (step * tau * Math.sqrt( c[6]  * X0[4]  + c[7] * X0[6]*X0[7] +c[8]*X0[8]  ));
//			return NormalDist.cdf01(zvalue);
//		case 7:
//			zvalue  = (X[j] - X0[j] - step * tau * (- c[7] * X0[6] * X0[7] +c[8] * X0[8] + c[10]*X0[9] - c[11] * X0[4] * X0[7] + c[12]*X0[10]+ c[13]*X0[10] ))
//			/ (step * tau * Math.sqrt( c[7] * X0[6] * X0[7] +c[8] * X0[8] + c[10]*X0[9] + c[11] * X0[4] * X0[7] + c[12]*X0[10]+ c[13]*X0[10]));
//			return NormalDist.cdf01(zvalue);
//			
//		case 8:
//			zvalue  = (X[j] - X0[j] - step * tau * ( c[7] * X0[6] * X0[7] -c[8] * X0[8] - c[9]*X0[8] ))
//			/ (step * tau * Math.sqrt( c[7] * X0[6] * X0[7] +c[8] * X0[8] + c[9]*X0[8]  ));
//			return NormalDist.cdf01(zvalue);
//		case 9:
//			zvalue  = (X[j] - X0[j] - step * tau * ( c[9] * X0[8] -c[10] * X0[9] ))
//			/ (step * tau * Math.sqrt(  c[9] * X0[8] +c[10] * X0[9]  ));
//			return NormalDist.cdf01(zvalue);
//		case 10:
//			zvalue  = (X[j] - X0[j] - step * tau * ( c[11] * X0[4] * X0[7] -c[12] * X0[10]-c[13] * X0[10] ))
//			/ (step * tau * Math.sqrt(  c[11] * X0[4] * X0[7] + c[12] * X0[10] + c[13] * X0[10] ));
//			return NormalDist.cdf01(zvalue);
//
//		default:
//			throw new IllegalArgumentException("Invalid state index");
//		}
		return NormalDist.cdf01( (X[j] - means[step][j]) / stdDevs[step][j] );
	}

	@Override
	public double getPerformance() {
//		return X[0];
//		return X[6];
//		return X[8];
		return X[9]; //P*
	}

}
