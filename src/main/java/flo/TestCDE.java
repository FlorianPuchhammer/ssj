package flo;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import umontreal.ssj.functionfit.LeastSquares;
import umontreal.ssj.hups.BakerTransformedPointSet;
import umontreal.ssj.hups.CachedPointSet;
import umontreal.ssj.hups.DigitalNetBase2;
import umontreal.ssj.hups.DigitalNetBase2FromFile;
import umontreal.ssj.hups.IndependentPointsCached;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.NestedUniformScrambling;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.RandomShift;
import umontreal.ssj.hups.Rank1Lattice;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.hups.StratifiedUnitCube;
import umontreal.ssj.latnetbuilder.DigitalNetBase2FromLatNetBuilder;
import umontreal.ssj.latnetbuilder.DigitalNetSearch;
import umontreal.ssj.latnetbuilder.Search;
import umontreal.ssj.mcqmctools.MonteCarloModelDouble;
import umontreal.ssj.mcqmctools.MonteCarloModelDoubleArray;
import umontreal.ssj.mcqmctools.RQMCExperiment;
import umontreal.ssj.mcqmctools.examples.AsianOptionVars;
import umontreal.ssj.mcqmctools.examples.BucklingStrengthVars;
import umontreal.ssj.mcqmctools.examples.LookBackOptionVars;
import umontreal.ssj.mcqmctools.examples.MultiNormalIndependent;
import umontreal.ssj.mcqmctools.examples.SanVars;
import umontreal.ssj.mcqmctools.examples.SanVarsCDE;
import umontreal.ssj.mcqmctools.examples.ShortColumnVars;
import umontreal.ssj.mcqmctools.examples.SumOfNormalsArray;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.randvar.NormalGen;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.PgfDataTable;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.stat.density.CDEAsianOption;
import umontreal.ssj.stat.density.CDEBucklingStrength;
import umontreal.ssj.stat.density.CDECantilever;
import umontreal.ssj.stat.density.CDENormalizedSumOfNormals;
import umontreal.ssj.stat.density.CDESan;
import umontreal.ssj.stat.density.CDEShortColumn;
import umontreal.ssj.stat.density.CDESum;
import umontreal.ssj.stat.density.ConditionalDensityEstimator;
import umontreal.ssj.stat.density.ConditionalDensityEstimatorDouble;
import umontreal.ssj.stat.density.DensityEstimator;
import umontreal.ssj.stat.density.LLBucklingStrength;
import umontreal.ssj.stat.density.LLCantilever;
import umontreal.ssj.stat.density.LLSan;
import umontreal.ssj.stat.density.LLShortColumn;
import umontreal.ssj.stat.density.LRLookBackOption;
import umontreal.ssj.stat.list.ListOfTallies;
import umontreal.ssj.stochprocess.BrownianMotion;
import umontreal.ssj.stochprocess.GeometricBrownianMotion;
import umontreal.ssj.util.Num;
import umontreal.ssj.util.PrintfFormat;

public class TestCDE {
	private double a, b;
	private boolean displayExec;
	private boolean producePlots;

	private double[] logIV;
	private double[] logN;
	private double logOfBase = Math.log(2.0);
	private double baseOfLog = 2.0;

	private String[] tableFields = { "logN", "logIV" };

	public TestCDE(double a, double b) {
		this.a = a;
		this.b = b;
	}

	private void preprocess(RQMCPointSet[] rqmcPts) {
		logN = new double[rqmcPts.length];
		for (int i = 0; i < rqmcPts.length; i++)
			logN[i] = Math.log((double) rqmcPts[i].getNumPoints()) / logOfBase;
		logIV = new double[logN.length];
	}

	public String formatHead(String pointLabel, String estimatorLabel, String numEvalPoints, int m) {
		StringBuffer sb = new StringBuffer("");
		sb.append("Estimation of IV rate over [" + a + ", " + b + "]\n");
		sb.append("----------------------------------------------------------------\n\n");
		sb.append("Estimator: " + estimatorLabel + "\n");
		sb.append("Point set used: " + pointLabel + "\n");
		sb.append("Number of repititions: m = " + m + "\n");
		sb.append("Evaluation points: " + numEvalPoints + "\n");
		sb.append("----------------------------------------------------------------\n\n");
		if (displayExec)
			System.out.print(sb.toString());
		return sb.toString();
	}

	public String estimateIVComputeTable(MonteCarloModelDoubleArray model, RQMCPointSet[] rqmcPts, int m,
			ConditionalDensityEstimator cde, double[] evalPts) {
		StringBuffer sb = new StringBuffer("");
		String str;

		str = "log(n)\t  log(IV)\n\n";

		sb.append(str);
		if (displayExec)
			System.out.print(str);

		double[][][] data;
		double[][] density;
		double[] variance;

//		ListOfTallies<Tally> statRepsList = new ListOfTallies<Tally>();
		ListOfTallies<Tally> statRepsList = ListOfTallies.createWithTally(model.getPerformanceDim());

		for (int i = 0; i < rqmcPts.length; i++) {// rqmc point sets indexed by i
			data = new double[m][rqmcPts[i].getNumPoints()][model.getPerformanceDim()];
			RQMCExperiment.simulReplicatesRQMC(model, rqmcPts[i], m, statRepsList, data);

			density = new double[m][evalPts.length];
			for (int rep = 0; rep < m; rep++) {
				density[rep] = cde.evalDensity(evalPts, data[rep]);
//				System.out.println("TEST:\t" + rep + "\t" + density[rep][0]);
			}
			variance = new double[evalPts.length];

			logIV[i] = Math.log(DensityEstimator.computeIV(density, a, b, variance)) / logOfBase;

			str = PrintfFormat.f(3, 1, logN[i]) + "\t " + PrintfFormat.f(8, 6, logIV[i]) + "\n";
//			str = PrintfFormat.f(3, 1, logN[i]) + "\t " + logIV[i] + "\n";

			sb.append(str);
			if (displayExec)
				System.out.print(str);
		}

		str = "\n\n";
		sb.append(str);
		if (displayExec)
			System.out.print(str);

		return sb.toString();
	}

	public String estimateIVComputeTable(MonteCarloModelDouble model, RQMCPointSet[] rqmcPts, int m,
			ConditionalDensityEstimatorDouble cde, double[] evalPts) {
		StringBuffer sb = new StringBuffer("");
		String str;

		str = "log(n)\t  log(IV)\n\n";

		sb.append(str);
		if (displayExec)
			System.out.print(str);

		double[][] data;
		double[][] density;
		double[] variance;

		Tally statReps = new Tally();

		for (int i = 0; i < rqmcPts.length; i++) {// rqmc point sets indexed by i
			data = new double[m][rqmcPts[i].getNumPoints()];
			RQMCExperiment.simulReplicatesRQMC(model, rqmcPts[i], m, statReps, data);

			density = new double[m][evalPts.length];
			for (int rep = 0; rep < m; rep++)
				density[rep] = cde.evalDensity(evalPts, data[rep]);

			variance = new double[evalPts.length];

			logIV[i] = Math.log(DensityEstimator.computeIV(density, a, b, variance)) / logOfBase;

			str = PrintfFormat.f(3, 1, logN[i]) + "\t " + PrintfFormat.f(8, 6, logIV[i]) + "\n";
			sb.append(str);
			if (displayExec)
				System.out.print(str);
		}

		str = "\n\n";
		sb.append(str);
		if (displayExec)
			System.out.print(str);

		return sb.toString();
	}

	public String estimateIVSlopes() {
		double[] regCoeffs = new double[2];

		String str = "Regression data:\n";
		str += "********************************************\n\n";
		str += "IV:\n";

		regCoeffs = LeastSquares.calcCoefficients(logN, logIV);
		str += "Slope:\t" + regCoeffs[1] + "\n";
		str += "Const.:\t" + regCoeffs[0] + "\n\n";

		if (displayExec)
			System.out.print(str);
		return str;
	}

	public String testIVRate(MonteCarloModelDoubleArray model, RQMCPointSet[] rqmcPts, int m,
			ConditionalDensityEstimator cde, double[] evalPoints) {
		StringBuffer sb = new StringBuffer("");
		sb.append(formatHead(rqmcPts[0].getLabel(), cde.toString(), Integer.toString(evalPoints.length), m));
		preprocess(rqmcPts);
		sb.append(estimateIVComputeTable(model, rqmcPts, m, cde, evalPoints));
		sb.append(estimateIVSlopes());

		return sb.toString();
	}

	public String testIVRate(MonteCarloModelDoubleArray model, ArrayList<RQMCPointSet[]> rqmcPtsList, int m,
			ConditionalDensityEstimator cde, double[] evalPoints) throws IOException {
		StringBuffer sb = new StringBuffer("");
		ArrayList<PgfDataTable> pgfTblList = new ArrayList<PgfDataTable>();
		for (RQMCPointSet[] rqmcPts : rqmcPtsList) {
			sb.append(testIVRate(model, rqmcPts, m, cde, evalPoints));
			if (producePlots)
				pgfTblList.add(genPgfDataTable(rqmcPts[0].getLabel(), rqmcPts[0].getLabel()));
		}
		if (producePlots)
			genPlots(cde.toString(), pgfTblList);
		return sb.toString();
	}

	public PgfDataTable genPgfDataTable(String tableName, String tableLabel) {
		int len = logN.length;
		double[][] pgfData = new double[len][tableFields.length];
		for (int i = 0; i < len; i++) {
			pgfData[i][0] = logN[i];
			pgfData[i][1] = logIV[i];
		}
		return new PgfDataTable(tableName, tableLabel, tableFields, pgfData);
	}

	public void genPlots(String cdeDescr, ArrayList<PgfDataTable> pgfTblList) throws IOException {
		FileWriter fw;
		String plotBody;

		plotBody = PgfDataTable.drawPgfPlotManyCurves("log(IV) vs log(n)", "axis", 0, 1, pgfTblList, (int) baseOfLog,
				"", " ");
		fw = new FileWriter(cdeDescr + "_IV.tex");
		fw.write(PgfDataTable.pgfplotFileHeader() + plotBody + PgfDataTable.pgfplotEndDocument());
		fw.close();
	}

	private static double[] genEvalPoints(int numPts, double a, double b, RandomStream stream) {
		double[] evalPts = new double[numPts];
		double invNumPts = 1.0 / ((double) numPts);
		for (int i = 0; i < numPts; i++)
			evalPts[i] = a + (b - a) * ((double) i + stream.nextDouble()) * invNumPts;
		return evalPts;
	}

	public boolean getDisplayExec() {
		return displayExec;
	}

	public void setDisplayExec(boolean displayExec) {
		this.displayExec = displayExec;
	}

	/**
	 * @return the producePlots
	 */
	public boolean getProducePlots() {
		return producePlots;
	}

	/**
	 * @param producePlots the producePlots to set
	 */
	public void setProducePlots(boolean producePlots) {
		this.producePlots = producePlots;
	}

	public double getLogOfBase() {
		return logOfBase;
	}

	public void setLogOfBase(double logOfBase) {
		this.logOfBase = logOfBase;
	}

	public double getBaseOfLog() {
		return baseOfLog;
	}

	public void setBaseOfLog(double baseOfLog) {
		this.baseOfLog = baseOfLog;
	}

	public static void main(String[] args) throws IOException {

		/*
		 * ************************ UTIL PARAMETERS
		 ****************************************/

		RandomStream noise = new MRG32k3a();
//		((MRG32k3a)noise).setSeed(new long[] {1606215560,1140697538,1523809004,1292913007,1858992010,665386629,1173670500});
		int mink = 14; // first log(N) considered
		int i;
		int m = 100; // m= 20;// Number of RQMC randomizations.
//		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152 }; // 13
//		int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 };
		int[] N = {16384, 32768, 65536, 131072, 262144, 524288};
//		int[] N = { 512, 1024, 2048, 4096, 8192, 16384 };
//		mink = 11;

//		int[] N = { 512, 1024, 2048, 4096, 8192, 16384 ,32768, 65536, 131072,262144};
		int numSets = N.length; // Number of sets in the series.
		int numEvalPts = 128; //numEvalPts = 8;// normal: 128
		
		//0.8^k
//		int[] aMult = {1, 103259, 511609, 131753, 107209, 228361, 192951, 467273, 151899, 97521, 143689, 239769, 286391, 258499, 252585, 92783, 56649, 190083, 75255, 304647};
//		int[] aMult = {1, 444567, 109925, 11927, 319113, 112579, 13599, 72667, 245697, 25001, 388683, 292851, 519755, 266677, 4533, 4533, 4533, 4533, 4533, 4533};
		
		//0.6^k
		int[] aMult = {1, 103259, 511609, 482163, 299529, 491333, 30987, 286121, 388189, 39885, 413851, 523765, 501705, 93009, 44163, 325229, 345483, 168873, 376109, 146111};

//		0.3^k
//		int[] aMult = {1, 444633, 249565, 81085, 453041, 223873, 16895, 383969, 437245, 61029, 341973, 156079, 167753, 440819, 520945, 183111, 8171, 459393, 438533, 119963};
		
//			int[][] aa = {
//				{ 1, 2433, 1715, 131, 3829, 2941, 395, 137, 659, 399, 137, 397, 397, 397, 397, 397, 397, 397, 397, 925,
//						925, 925, 925, 3039, 3039 }, // 13
//				{ 1, 6915, 3959, 7595, 6297, 1183, 1545, 4297, 5855, 869, 7413, 7413, 7413, 7413, 7413, 7413, 7413,
//						7413, 7413, 7413, 7413, 7413, 7413, 7413, 7413 }, // 14
//				{ 1, 12033, 7503, 15835, 1731, 273, 12823, 7895, 16313, 1591, 8571, 16313, 16313, 16313, 16313, 16313,
//						16313, 16313, 16313, 16313, 16313, 16313, 16313, 16313, 16313 }, // 15
//
//				{ 1, 25015, 5425, 24095, 30915, 12607, 29583, 1203, 10029, 23717, 21641, 21381, 21381, 21381, 21381,
//						21381, 21381, 21381, 21381, 21381, 21381, 21381, 21381, 21381, 21381 }, // 16
//
//				{ 1, 50687, 44805, 12937, 21433, 42925, 47259, 14741, 265, 60873, 28953, 36059, 25343, 36059, 36059,
//						36059, 36059, 36059, 36059, 36059, 36059, 36059, 36059, 36059, 36059 }, // 17
//
//				{ 1, 100135, 28235, 39865, 43103, 121135, 93235, 1647, 50163, 39377, 122609, 115371, 89179, 69305,
//						89179, 89179, 89179, 89179, 89179, 89179, 89179, 89179, 89179, 89179, 89179 }, // 18
//
//				{ 1, 154805, 242105, 171449, 27859, 174391, 129075, 50511, 24671, 156015, 5649, 194995, 71129, 71127,
//						71129, 71129, 71129, 71129, 71129, 71129, 71129, 71129, 71129, 71129, 71129 }, // 19
//
//				{ 1, 387275, 314993, 50301, 174023, 354905, 481763, 269925, 287657, 445979, 109871, 314929, 215641,
//						166525, 184945, 184945, 184945, 184945, 184945, 184945, 184945, 184945, 184945, 184945, 184945 } // 20
//
//		};

		//0.6^k
//				int[][] aa = {
//						{ 1, 3455, 1967, 1029, 2117, 3871, 533, 2411, 1277, 2435, 1723, 3803, 1469, 569, 1035, 3977, 721, 797, 297, 1659 }, // 13
//
//						{ 1, 6915, 3959, 7743, 3087, 5281, 6757, 3369, 7107, 6405, 7753, 1641, 3613, 1819, 5827, 2087, 4417, 6909, 5623, 4739 }, // 14
//
//						{ 1, 12031, 14297, 677, 6719, 15787, 10149, 7665, 1017, 2251, 12105, 2149, 16273, 14137, 8179, 6461, 15051, 6593, 12763, 8497 }, // 15
//
//						{ 1, 19463, 8279, 14631, 12629, 26571, 30383, 1337, 6431, 3901, 12399, 20871, 5175, 3111, 26857, 15111, 22307, 30815, 25901, 27415 }, // 16
//
//						{ 1, 38401, 59817, 33763, 32385, 2887, 45473, 48221, 3193, 63355, 40783, 37741, 54515, 11741, 10889, 17759, 6115, 18687, 19665, 26557}, // 17
//
//						{ 1, 100135, 28235, 46895, 82781, 36145, 36833, 130557, 73161, 2259, 3769, 2379, 80685, 127279, 45979, 66891, 8969, 56169, 92713, 67743 }, // 18
//
//						{ 1, 154805, 242105, 171449, 27859, 76855, 183825, 38785, 178577, 18925, 260553, 130473, 258343, 79593, 96263, 36291, 2035, 198019, 15473, 148703 }, // 19
//
//						{1, 387275, 314993, 50301, 174023, 354905, 303021, 486111, 286797, 463237, 211171, 216757, 29831, 155061, 315509, 193933, 129563, 276501, 395079, 139111 } // 20
//				};
				
//				//1,0.8,0.5,0.6^k
//				int[][] aa = {
//						{1 , 2433, 3867, 283, 207, 1349, 1671, 3191, 1193, 767, 3561, 1483, 3459, 227, 3487, 1571, 1451, 3245, 3697, 1713}, // 13
//
//						{ 1, 6915, 3959, 3677, 4185, 1881, 3553, 771, 2611, 7885, 4279, 6321, 1507, 1613, 6629, 5895, 885, 801, 7279, 4529}, // 14
//
//						{ 1, 12033, 7503, 15835, 11159, 6843, 1599, 9731, 14143, 10099, 11799, 13157, 8319, 3323, 16135, 11505, 7267, 4113, 7141, 2143 }, // 15
//
//						{ 1, 25015, 5425, 26845, 21115, 20205, 26305, 21061, 11223, 25775, 29995, 1225, 30373, 21537, 15751, 7781, 7195, 18091, 3449, 2025 }, // 16
//
//						{ 1, 38401, 59817, 33763, 32385, 36297, 42029, 44775, 45599, 54653, 61215, 1687, 42611, 8195, 49903, 399, 58607, 17713, 28367, 64845}, // 17
//
//						{ 1, 100135, 28235, 46895, 16787, 74245, 29079, 126763, 118855, 10467, 102259, 130375, 74615, 129291, 3393, 111617, 8725, 29261, 123261, 27317 }, // 18
//
//						{ 1, 216675, 59253, 23017, 150663, 58193, 211521, 221789, 202545, 141437, 26489, 81503, 83939, 187207, 121709, 162983, 249565, 16241, 76505, 225851}, // 19
//
//						{1, 443165, 95693, 34519, 391949, 437359, 16131, 129289, 399557, 153237, 248855, 104805, 12273, 354063, 169257, 154189, 246051, 168245, 99617, 143389} // 20
//				};
				
				//0.05^k
				int[][] aa = {
						{1, 3455, 1899, 2921, 3663, 2823, 3977, 2761, 255, 845, 3029, 3831, 2089, 3691, 1771, 3907, 337, 3735, 1373, 1795}, // 13

						{ 1, 6915, 4877, 7479, 1203, 3941, 2159, 3225, 5219, 6307, 2643, 633, 7139, 869, 7239, 7019, 8151, 3853, 8019, 5731}, // 14

						{ 1, 12033, 3801, 5023, 10647, 14127, 12751, 7461, 11901, 1167, 14349, 1951, 2209, 7397, 2505, 5675, 12195, 1801, 7707, 13443 }, // 15

						{ 1, 25015, 11675, 7425, 3289, 17821, 5649, 32161, 10285, 12031, 26337, 13403, 14547, 18661, 7993, 1299, 15111, 12735, 13129, 12655 }, // 16

						{ 1, 38401, 48799, 17301, 59639, 20297, 26805, 53109, 4365, 14055, 5023, 48499, 37937, 5155, 44255, 61671, 11409, 38529, 61887, 19183}, // 17

						{ 1, 96407, 36479, 31333, 63411, 80945, 24597, 41083, 70179, 42983, 62013, 48035, 80011, 105415, 108151, 68869, 104973, 20719, 72257, 59193 }, // 18

						{ 1, 154805, 243089, 211205, 258913, 18107, 174117, 67287, 3585, 155767, 31401, 154275, 35513, 36509, 162377, 51021, 88413, 190981, 145989, 257551}, // 19

						{1, 387275, 457903, 282967, 117983, 355873, 439959, 109733, 382437, 297385, 267803, 68841, 343399, 171303, 420841, 136437, 423733, 355591, 415917, 406205} // 20
				};
		
		//0.3^k
//		int[][] aa = {
//				{ 1, 2431, 3739, 519, 1183, 2997, 587, 3699, 1955, 4027, 3627, 2573, 2823, 2837, 2563, 3183, 1477, 61, 1643, 2943}, // 13
//
//				{ 1, 6915, 3959, 6333, 7999, 2837, 4565, 253, 4797, 3765, 5589, 5327, 1401, 2105, 1105, 3445, 7853, 2261, 2851, 7981 }, // 14
//
//				{ 1, 12033, 3801, 12891, 11071, 8875, 12759, 359, 16037, 4747, 12989, 12051, 9369, 12917, 13271, 11677, 9521, 10517, 5653, 11615}, // 15
//
//				{ 1, 25015, 11675, 7425, 13211, 31259, 28561, 9081, 28115, 10211, 21197, 15587, 30379, 22987, 4875, 14081, 21883, 4895, 25157, 26999 }, // 16
//
//				{ 1, 38401, 29759, 57041, 15511, 22291, 55807, 6139, 11933, 51447, 61827, 17569, 18153, 20495, 14457, 21833, 24791, 23413, 65101, 53801}, // 17
//
//				{ 1, 100135, 28235, 46895, 111521, 92189, 50261, 130597, 40617, 7843, 77325, 90483, 1043, 28951, 105595, 105679, 110293, 66221, 99109, 44845 }, // 18
//
//				{1, 216675, 236269, 124797, 117397, 103059, 204801, 64279, 24525, 44767, 182765, 5309, 198791, 113495, 131941, 178935, 203931, 11411, 74323, 92107 }, // 19
//
//				{1, 443165, 71715, 105083, 230107, 501113, 73125, 207581, 91007, 189617, 174013, 514245, 134005, 101019, 448739, 159237, 447629, 509485, 47631, 347173} // 20
//		};
		
		//rand-CBC, 0.8^k for k<=5. Default 0.005
//		int[][] aa = {
//				{1, 2433, 1715, 131, 3829, 77, 2941, 451, 301, 3555, 715, 647, 1687, 3555, 3807, 3491, 2289, 3555, 2629, 1321}, // 13
//
//				{ 1, 6915, 3959, 7595, 6297, 5539, 2911, 3815, 1445, 4041, 6351, 7567, 955, 3373, 4897, 1, 2895, 3159, 3705, 1}, // 14
//
//				{ 1, 12033, 7503, 15835, 1731, 7355, 14207, 13803, 5439, 14777, 4495, 12343, 8457, 8515, 2641, 2899, 10967, 7503, 2991, 14159 }, // 15
//
//				{1, 25015, 5425, 24095, 30915, 31459, 22671, 3241, 27623, 2507, 375, 773, 28075, 16715, 24189, 8503, 6139, 19989, 18229, 16367}, // 16
//
//				{ 1, 38401, 59817, 33763, 32385, 25683, 19031, 16843, 56259, 45715, 10647, 29487, 33613, 54191, 61627, 8179, 28923, 41323, 23889, 51833}, // 17
//
//				{ 1, 100135, 28235, 39865, 43103, 43875, 13455, 30249, 122413, 40425, 118355, 15723, 36781, 28745, 61213, 18759, 79909, 94625, 79717, 8149}, // 18
//
//				{ 1, 216675, 59253, 238453, 221081, 1791, 112551, 201557, 186837, 67483, 49069, 115325, 57611, 156083, 120157, 214503, 64789, 3225, 177271, 55507}, // 19
//
//				{1, 443165, 95693, 34519, 235147, 15163, 321133, 38819, 220363, 191491, 513877, 452379, 326861, 425819, 400443, 487677, 39353, 221815, 521941, 343743} // 20
//		};
		
		//0.9^k
//		int[][] aa = {
//				{1, 1235, 3653, 2741, 4011, 149}, // 13
//
//				{ 1, 6957, 3653, 2741, 4181, 8043}, // 14
//
//				{1, 6957, 3653, 13643, 12203, 8341 }, // 15
//
//				{1, 25811, 29115, 13643, 20565, 16533 }, // 16
//
//				{1, 25811, 29115, 13643, 20565, 49003}, // 17
//
//				{1, 25811, 101957, 13643, 20565, 82069 }, // 18
//
//				{1, 236333, 160187, 117429, 241579, 82069}, // 19
//
//				{1, 236333, 160187, 117429, 282709, 82069} // 20
//		};
		
		//1.0^k
//				int[][] aa = {
//						{1, 1235, 581, 2741, 85, 21}, // 13
//
//						{ 1, 4955, 3705, 681, 7479, 1655}, // 14
//
//						{ 1, 4955, 12679, 681, 8905, 1655 }, // 15
//
//						{1, 25811, 29115, 13643, 20565, 12267 }, // 16
//
//						{1, 25811, 29115, 13643, 20565, 53269}, // 17
//
//						{1, 103259, 12679, 681, 107209, 67191 }, // 18
//
//						{1, 236333, 160187, 248501, 20565, 184341}, // 19
//
//						{1, 103259, 511609, 262825, 369353, 67191} // 20
//				};
		
		/*
		 * ************************ MODEL
		 ****************************************/
		// LOOKBACK OPTION
		
		int dim = 12;
		MonteCarloModelDoubleArray model = new LookBackOptionVars(dim);
				
				
//		double[] mus = {2.9E7,500.0,1000.0}; //Canti
//		double[] sigmas = {1.45E6,100.0,100.0}; //Canti
//		int dim = mus.length;

		// CANTI, NORMALS
//		int dim = 11;
//		double[] mus = new double[dim];
//		Arrays.fill(mus,0.0);
//		double[] sigmas = new double[dim];
////		Arrays.fill(sigmas,1.0);
//		sigmas[0] = 1.0;
//		for(int j = 1; j<dim; j++)
//			sigmas[j] = sigmas[j-1] / Math.sqrt(2.0);
//
//		MonteCarloModelDoubleArray model = new MultiNormalIndependent(mus,sigmas);
	
		

		// BUCKLING
//		double[] mus = { 0.992 * 24.0, 1.05 * 0.5, 1.3 * 34.0, 0.987 * 29.0E3,0.35, 5.25};
//		double[] covs = { 0.028, 0.044, 0.1235, 0.076,0.05,0.07 };
//		int dim = mus.length;
//		double[] sigmas = new double[dim];
//		for(int j = 0; j < dim; j++)
//			sigmas[j] = mus[j] * covs[j];
//		MonteCarloModelDoubleArray model = new BucklingStrengthVars(mus, sigmas);

		// SAN
//		int dim = 13;

//		MonteCarloModelDoubleArray model = new SanVars("san13a.dat"); //LL

//		MonteCarloModelDoubleArray model = new SanVarsCDE("san13a.dat"); //CDE

		// SHORT COLUMN
//		double h = 15.0;
//		double bb = 5.0;
//		double muY = 5.0;
//		double muM = 2000.0;
//		double muP = 500.0;
//		double sigmaY = 0.5;
//		double[][] sigma = { { 400.0, 50.0 }, { 0.0, 86.60254037844386 } };
//
//		MonteCarloModelDoubleArray model = new ShortColumnVars(muY, muM, muP, sigmaY, sigma);
//		int dim = 3;
		
		//ASIAN GBM
//		double strike= 101.0;
//		double s0 = 100.0;
//		double sigma = 0.12136;
//		int dim =12;
//		double[] obsTimes = new double[dim + 1];
//		obsTimes[0] = 0.0;
//		for (int j = 1; j <= dim; j++) {
//			obsTimes[j] = (double) j / (double) dim;
//		}
//		double r =Math.log(1.09); r = 0.0;
//		AsianOptionVars model = new AsianOptionVars(r,dim, obsTimes,strike);
//		NormalGen gen = new NormalGen(new MRG32k3a());
//		GeometricBrownianMotion sp = new GeometricBrownianMotion(s0, 0.1,
//				sigma, new BrownianMotion(0.0, 0.0, 1.0, gen));
//		model.setProcess(sp);
		
		//SUM OF NORMALS
//		int dim = 10;
//		double[] mu = new double[dim];
//		Arrays.fill(mu,0.0);
//		double[] sigma = new double[dim];
//		sigma[0] = 1.0;
//		double sigmaHidden = 1.0;
//		double temp = 1.0;
//		int leave = 9;
//		int index = 0;
//		for(int j = 0; j < dim+1; ++j) {
//			
//			if(j!=leave) {
//				if(j!=0)
//					sigma[index]=temp;
//				index++;
//			}
//			else {
//				if(j!=0) 
//					sigmaHidden = temp;	
//		
//					
//			}
////			temp *= Math.sqrt(2.0);
//			temp *= 2.0;
////			temp*=1.0;
//		}
//		
//		String str1 = "SIGMA: ";
//		for(double s : sigma)
//			str1 += s + ", "; 
//		str1+= "SIGMA HIDDEN: "+ sigmaHidden;
//		System.out.println(str1);
//		
//		SumOfNormalsArray model = new SumOfNormalsArray(mu,sigma);

		/*
		 * ************************ DENSITY ESTIMATOR
		 ****************************************/

		double strike= 101.0;
		double s0 = 100.0;
		double sigma = 0.12136;
		double r = 0.1;
		
		double a = strike; double b = strike + 34.4; //Lookback; cuts 0.08 left and 0.05 right --> 87% of mass
		
		ConditionalDensityEstimator cde = new LRLookBackOption(a,b,dim,s0,strike,r,sigma);
		String descr = "LRLookBack";
		
		//CANTILEVER
//		double a = 0.407;
//		double b = 1.515;	
//		double D0 = 2.2535;
//		a = (a+1) * D0;
//		b = (b+1) * D0;
//		
//		double L = 100.0;
//		double t = 2.0;
//		double w = 4.0;
////		double[] weights = {0.25639625873702715, 8.639295478835745E-7 , 0.743602877333425};
////		double[] weights = {0.25, -1.0 , 0.75};
//		double[] weights = {-1.0,1.0, -1.0};
//
//		ConditionalDensityEstimator cde = new CDECantilever(L, t, w, mus[0], sigmas[0], mus[1], sigmas[1], mus[2],
//				sigmas[2], weights);
//		String descr = "cdeCantiY2";

//		double pp = 0.76171875;

//		ConditionalDensityEstimator cde = new LLCantilever(L, t, w, mus[0],  sigmas[0],  mus[1],  sigmas[1], mus[2],  sigmas[2],pp);
//		String descr = "llCantiOpt";

		// NORMALS
//		double a = -2.0;
//		double b = 2.0;
//		
//		double[] weights = new double[dim];
////		Arrays.fill(weights,1.0/((double) dim)); //set the unused ones negative, then they will be omitted!
//		
//		Arrays.fill(weights,-1.0); //set the unused ones negative, then they will be omitted!
//		weights[0] = 1.0;
//		
//		ConditionalDensityEstimator cde = new CDENormalizedSumOfNormals(dim,weights,sigmas);
//		String descr = "cdeSumOfNormals" + dim;

		// BUCKLING
//		double a = 0.5169;
//		double b = 0.6511;
//
////		double pp = 0.0; //LL guess
////		double pp = 0.146171875; //ll opt
////		double pp = 0.1457903184811636; //ll fit
//
////		ConditionalDensityEstimator cde = new LLBucklingStrength(mus[4],sigmas[4],mus[5],sigmas[5],pp);
////		String descr = "LLBucklingStrengthFit146";
////		
//
//		double pp = 0.0; //CDE guess
////		double pp = 0.00244140625; //CDE opt
////		double pp = 0.002378784944706333; // CDE quad fit
//
//		ConditionalDensityEstimator cde = new CDEBucklingStrength(mus[4],sigmas[4],mus[5],sigmas[5],pp);
//		String descr = "CDEBucklingStrengthG5";

		// SAN
//		double a =22.0;
//		double b = 106.24; //95% non-centralized
//
////		ConditionalDensityEstimator cde = new LLSan();
////		String descr = "LLSan";
//
//		ConditionalDensityEstimator cde = new CDESan("san13a.dat");
//		String descr = "CDESan";

		// SHORT COLUMN
//		double a = -5.338; // ShortColumn
//		double b = -0.528; // 99% centered
//
//		ConditionalDensityEstimator cde = new CDEShortColumn(bb,h,muY,sigmaY);
//		ConditionalDensityEstimator cde = new LLShortColumn(bb, h, muY, sigmaY, 0.5);
//		String descr = "LRSHortColumn";
//		String descr = "CDEShortColumn";

		//ASIAN GBM
		
		
		
//		double a = 0.0; double b=27.13; 
//		double [] weights = new double[dim];
//		double norma = 0.0;
//		for(int j = 0; j < dim; j++) {
//			weights[j] = 1.0/(double)((j+1.0));
//			weights[j] = (double) 1.0;
//			weights[j] = (double) ((j+1.0));
//			weights[j] = (double) ((j+1.0) * (j+1.0));
//			norma += weights[j];
//		}
//		for(int j = 0; j < dim; j++) {
//			weights[j] /= norma;
//			System.out.println(j + ", " + weights[j]);
//		}
		
//		Arrays.fill(weights,0.0);
//		weights[dim-1]= 1.0;
//		
//		ConditionalDensityEstimator cde = new CDEAsianOption(a,b, dim,
//				s0, strike,sigma,r,weights);
//		String descr = "CDEAsianOption";
		
		//SUM OF NORMALS
		
//		double a = -2.0; 
//		double b=2.0; 
//		
//		
//		double norma = 0.0;
//		
//		for(double s:sigma)
//			norma += s*s;
//		norma += sigmaHidden * sigmaHidden;
//		norma = Math.sqrt(norma);
//		
////		a-=norma; b+=norma;
////		norma=1.0;
//		
//		System.out.println("NORMA = " + norma);
//		NormalDist dist = new NormalDist(0.0,sigmaHidden);
//		ConditionalDensityEstimator cde = new CDESum(dist,norma);
//		String descr = "CDESumOfNormals";
		
		
		double[] evalPoints = genEvalPoints(numEvalPts, a, b, noise);
		
		/*
		 * ************************ POINT SETS
		 ****************************************/
		// Create a list of series of RQMC point sets.
		ArrayList<RQMCPointSet[]> listRQMC = new ArrayList<RQMCPointSet[]>();
		PointSet p;
		PointSetRandomization rand;
		RQMCPointSet[] rqmcPts;
		
		// Independent points (Monte Carlo)
		rqmcPts = new RQMCPointSet[numSets];
		for (i = 0; i < numSets; ++i) {
			p = new IndependentPointsCached(N[i], dim);
			rand = new RandomShift(noise);
			rqmcPts[i] = new RQMCPointSet(p, rand);
		}
		rqmcPts[0].setLabel("Independent points");
		listRQMC.add(rqmcPts);
//		
//		// Stratification
//		rqmcPts = new RQMCPointSet[numSets];
//		int k;
//		for (i = 0; i < numSets; ++i) {
//			k = (int) Math.round(Math.pow(Num.TWOEXP[i + mink], 1.0 / (double) (dim)));
//			p = new StratifiedUnitCube(k, dim);
//
//			rand = new RandomShift(noise);
//			rqmcPts[i] = new RQMCPointSet(p, rand);
//		}
//		rqmcPts[0].setLabel("Stratification");
//		listRQMC.add(rqmcPts);
		
//		 //lattice+shift
//		 rqmcPts = new RQMCPointSet[numSets];
//		 for (i = 0; i < numSets; ++i) {
//		
////		 p = new Rank1Lattice(N[i],aa[mink-13+i],dim);
//			 p = new Rank1Lattice(N[i],aMult,dim);
//
//		
//		 rand = new RandomShift(noise);
//		 rqmcPts[i] = new RQMCPointSet(p, rand);
//		 }
//		 rqmcPts[0].setLabel("Lattice+Shift");
//		 listRQMC.add(rqmcPts);
//		
//		// lattice+baker
//				 rqmcPts = new RQMCPointSet[numSets];
//				 for (i = 0; i < numSets; ++i) {
//				
//				 p =  new BakerTransformedPointSet(new Rank1Lattice(N[i],aa[mink-13+i],dim));
////					 p =  new BakerTransformedPointSet(new Rank1Lattice(N[i],aMult,dim));
//
//				
//				 rand = new RandomShift(noise);
//				 rqmcPts[i] = new RQMCPointSet(p, rand);
//				 }
//				 rqmcPts[0].setLabel("Lattice+Baker");
//				 listRQMC.add(rqmcPts);
		 
//		 Sobol + LMS
//		rqmcPts = new RQMCPointSet[numSets];
//		for (i = 0; i < numSets; ++i) {
//
//			p = new SobolSequence(i + mink, 31, dim);
//
//			rand = new LMScrambleShift(noise);
//			rqmcPts[i] = new RQMCPointSet(p, rand);
//		}
//		rqmcPts[0].setLabel("Sobol+LMS");
//		listRQMC.add(rqmcPts);
//
//		// Sobol+NUS
//		rqmcPts = new RQMCPointSet[numSets];
//		for (i = 0; i < numSets; ++i) {
//			CachedPointSet cp = new CachedPointSet(new SobolSequence(i + mink, 31, dim));
//			cp.setRandomizeParent(false);
//			p = cp;
//
//			rand = new NestedUniformScrambling(noise);
//			rqmcPts[i] = new RQMCPointSet(p, rand);
//		}
//		rqmcPts[0].setLabel("Sobol+NUS");
//		listRQMC.add(rqmcPts);

//					// Poly Lat + LMS
//					rqmcPts = new RQMCPointSet[numSets];
//					for (i = 0; i < numSets; ++i) {
//						p = new DigitalNetBase2FromLatNetBuilder("/u/puchhamf/misc/latnetbuilder/output/poly/interlaced4/2^" +(mink+i) + "/outputMachine.txt",mink+i,31,dim);
//			
//						rand = new LMScrambleShift(noise);
//						rqmcPts[i] = new RQMCPointSet(p, rand);
////						System.out.println("TEST:\n" +((DigitalNetBase2FromLatNetBuilder) p).toStringDetailed());
////						System.out.println("TEST:\n" + pp.genMat.length);
//
//					}
//					rqmcPts[0].setLabel("SobT+LMS");
//					listRQMC.add(rqmcPts);
//					
//					for(i=0; i < numSets; ++i) {
////						System.out.println("TEST:\n" +((DigitalNetBase2FromLatNetBuilder) rqmcPts[i].getPointSet()).toStringDetailed());
//					}
//						
//					
//					// PolyLat+NUS
//					rqmcPts = new RQMCPointSet[numSets];
//					for (i = 0; i < numSets; ++i) {
//						CachedPointSet cp = new CachedPointSet(new DigitalNetBase2FromLatNetBuilder("/u/puchhamf/misc/latnetbuilder/poly/digNet/interlaced4/2^" +(mink+i) + "/outputMachine.txt",mink+i,31,dim));
//						cp.setRandomizeParent(false);
//						p = cp;
//			
//						rand = new NestedUniformScrambling(noise);
//						rqmcPts[i] = new RQMCPointSet(p, rand);
//					}
//					rqmcPts[0].setLabel("SobT+NUS");
//					listRQMC.add(rqmcPts);





		TestCDE test = new TestCDE(a, b);

		test.setDisplayExec(true);
		test.setProducePlots(true);

		FileWriter fw = new FileWriter(descr + ".txt");
		String str = test.testIVRate(model, listRQMC, m, cde, evalPoints);

		fw.write(str);
		fw.close();

		System.out.println(str);
	}

}
