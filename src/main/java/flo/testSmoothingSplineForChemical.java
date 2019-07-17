package flo;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import umontreal.ssj.functionfit.SmoothingCubicSpline;
import umontreal.ssj.stat.PgfDataTable;

public class testSmoothingSplineForChemical {
	public static void main(String[] args) throws IOException {

		String filepath = "/u/puchhamf/misc/jars/chemical/PKA/PKA/fit/20/smoothedCoeffs/";

		double[] X = new double[] { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14. };
	
		/*
		 * 
		 * SCHLOEGL
		 * 
		 */
		// b0
//		String coeffName = "b0";
//		double[] Y = new double[] {66942.9, 20493.2, -4032.8, -22563.6, -22119.7, -18233.7, -12526.3, 
//			-8703.65, -6003.77, -3821.46, -2555.56, -1524.96, -664.31, 54.7635};
		// b1
//		String coeffName = "b1";
//		double[] Y = new double[] {-456.194, -149.023, 30.0941, 169.015, 160.567, 130.008, 86.466, 
//				59.7229, 41.5349, 26.8114, 19.1004, 12.4798, 6.4394, 0.380509};
		// b2
//		String coeffName = "b2";
//		double[] Y = new double[] { -0.631591, -0.184944, 0.0506769, 0.231563, 0.225048, 0.185245, 0.127592, 0.0890393,
//				0.0618056, 0.0397661, 0.0269221, 0.0164027, 0.00753944, -9.27562E-6 };
//		// b3
//		String coeffName = "b3";
//		double[] Y = new double[] { 0.730923, 0.251439, -0.0750055, -0.316892, -0.286371, -0.225153, -0.147931,
//				-0.104164, -0.0745831, -0.0499204, -0.0364048, -0.0242956, -0.0126206, -0.00055135 };
		// b4
//		String coeffName = "b4";
//		double[] Y = new double[] { 0.00407109, 0.00122252, -0.000441922, -0.00176654, -0.00165127, -0.00133131,
//				-0.000888246, -0.000616594, -0.00043179, -0.000282034, -0.00020246, -0.000133328, -0.0000692561,
//				-3.21771E-6 };
//		// b5
//		String coeffName = "b5";
//		double[] Y = new double[] { -0.000270612, -0.000149197, -0.0000778334, -0.0000388924, -0.0000221292,
//				-0.0000149312, -0.000012279, -0.0000107994, -9.86091E-6, -9.24329E-6, -8.37341E-6, -7.4059E-6,
//				-6.23209E-6, -4.40913E-6 };
		// b6
//				String coeffName = "b6";
//				double[] Y = new double[] {-5.23471E-6, -1.35053E-6, 1.38263E-6, 3.51508E-6, 
//						3.07746E-6, 2.40466E-6, 1.60519E-6, 1.15259E-6, 
//						8.46774E-7, 5.92233E-7, 4.47848E-7, 3.16071E-7, 
//						1.86068E-7, 4.55399E-8};
		
		/*
		 * 
		 * PKA
		 * 
		 */
		// b0
//		String coeffName = "b0";
//		double[] Y = new double[] {2.6810988477686408E8, -6.748327067055596E7, -1.9139867406405274E7, -3.8922397933225036E7, 3365664.8631102773, 8911095.65551753, -3.193391140626513E7, 
//				-1.0712566741711138E7, -9074798.699787972, 4.1227453973241396E7, 3.8544615755171806E7, 4.283425634504837E7, 1.6999693906417727E7, -3885730.5643284693};
		// b1
//		String coeffName = "b1";
//		double[] Y = new double[] {-7293.467894847656, 2369.366232311366, 553.1537621973335, 1238.4648270338182, -211.98463177491206, -366.6616345884858, 1137.5994838319978, 
//				340.70768430752605, 334.34372487661864, -1632.8766946488759, -1542.3612019002755,-1694.0848111213386, -672.8666443414934, 152.05610261160814};
		// b2
//		String coeffName = "b2";
//		double[] Y = new double[] { -7670.4113031584, 1546.2444399082508, 525.0958244664637, 985.4454200241116, -9.561394606244653, -166.97106031582953, 721.3292003350722, 273.32187709907873,
//				199.0092782603552, -812.1917155963882, -748.1521498658192,-846.3784068484074, -335.2874191866059, 78.51337774823585};
//		// b3
//		String coeffName = "b3";
//		double[] Y = new double[] {-8073.197419929038,715.5418081853096, 498.2538129487588, 732.4087163808489, 196.4796990861376, 40.79364469426986, 305.7553122715082,
//				202.04292369003753, 59.72733271575888, 10.929166247123407, 48.723890111313594, 5.979775474037456, 5.377442540532549, 6.433354089820613};
		// b4
//		String coeffName = "b4";
//		double[] Y = new double[] { -0.0068730895625955775, -0.015356414092975812,-5.248200114554972E-4, -0.004824720273115525, 0.0038691508558502206, 0.0037173386234362425,
//				-0.008067590052351912, -0.0011395923410058926, -0.0024447014010362963, 0.016009371263993634, 0.015494849389070713, 0.01645490369477779, 0.006530410757118996,
//				-0.0014515539082487188};
//		// b5
//		String coeffName = "b5";
//		double[] Y = new double[] { -0.004622113816324837, -0.008166591173405208, -2.3582369983299084E-4, -0.0024882798217307988, 0.0021888593350479664,
//				0.002616678415710106, -0.004064828846605848, -0.0011974489758679505, -0.0019142403956777184, 0.008221280527475242, 0.008015229189028996, 0.008854411359959704,
//				0.003646221591915652,-6.364926507904067E-4};
		// b6
//		String coeffName = "b6";
//		double[] Y = new double[] { -4.21090307342876, 0.3751670479788333,0.2619025548104334, 0.38696380490818566, 0.10459575920940485, 0.021721358253140748,
//				0.1629500037773174,0.10794600536281111, 0.031973930566861763, 0.005843908253657647, 0.026302470100021177, 0.003195650355293847,	0.0029134978614727292, 0.00352596174230501};
		
		// b7
//		String coeffName = "b7";
//		double[] Y = new double[] {3.694792799498078E-8, 9.790384996429959E-8, 3.6814234345117525E-9, 3.4121527765144116E-8, -2.6273757479937983E-8, -1.7838744966098648E-8, 6.271928741596924E-8, 
//				-3.2270103823063905E-9, 4.840564221876551E-9, -1.2755189595712674E-7, -1.2289393307229197E-7, -1.2172862932027393E-7, -4.447744057671139E-8, 1.5152777773690173E-8};
		
		// b8
//		String coeffName = "b8";
//		double[] Y = new double[] {4.5024930220206754E-8,6.32977981414732E-8, 1.0025534169913739E-9, 2.0387440061681466E-8, -2.4941107998200313E-8, -4.8997470658027745E-8, 3.3410174605375634E-8, 
//				3.9104413532731704E-8, 5.1213286316404064E-8, -7.97488000170376E-8, -8.266005025393289E-8, -1.1444942450525568E-7, -5.688145501359465E-8, -9.578188435933273E-10};
		
		// b9
		String coeffName = "b9";
		double[] Y = new double[] {9.624647621090895E-4, -8.63822048525019E-5, -6.0516541245859134E-5, -9.00641941068571E-5, -2.4605167208305113E-5, -5.11249451536174E-6,	-3.8394725583492715E-5, 
				-2.5525462444573476E-5, -7.5828756070796575E-6,-1.3841186614355474E-6, -6.311660688770666E-6, -7.555276608213138E-7, -7.024312824051466E-7, -8.642516334468992E-7};

		int n = X.length;

		double[] rhos = { 0.2, 0.4, 0.6, 0.8 };
		String rhoName;
		SmoothingCubicSpline fit;
		int m = 50;
		double[] Xp = new double[m + 1]; // Xp, Yp are spline points
		double[] Yp = new double[m + 1];
		double h = (X[n - 1] - X[0]) / m; // step
		StringBuffer sb = new StringBuffer("");
		double[][] plotData = new double[Xp.length][];
		FileWriter fw;
		for (double rho : rhos) {
			sb.setLength(0);
			rhoName = "0" + (int) (rho * 10);

			fit = new SmoothingCubicSpline(X, Y, rho);

			for (int i = 0; i <= m; i++) {
				double z = X[0] + i * h;
				Xp[i] = z;
				Yp[i] = fit.evaluate(z); // evaluate spline at z
			}

			for (int i = 0; i < Xp.length; i++) {
				plotData[i] = new double[2];
				plotData[i][0] = Xp[i];
				plotData[i][1] = Yp[i];
			}
			PgfDataTable table = new PgfDataTable("Fit", "fit", new String[] { "x", "y" }, plotData);

			double[][] plotData2 = new double[X.length][];
			for (int i = 0; i < X.length; i++) {
				plotData2[i] = new double[2];
				plotData2[i][0] = X[i];
				plotData2[i][1] = Y[i];
			}
			PgfDataTable table2 = new PgfDataTable("Fit", "data", new String[] { "x", "y" }, plotData2);

			ArrayList<PgfDataTable> pgfTblList = new ArrayList<PgfDataTable>();
			pgfTblList.add(table);
			pgfTblList.add(table2);
			
			sb.append(PgfDataTable.pgfplotFileHeader());
//			sb.append(table.drawPgfPlotSingleCurve("Fit", "axis", 0, 1, 2, "", ""));
//			sb.append(table2.drawPgfPlotSingleCurve("Fit", "axis", 0, 1, 2, "", ""));
			sb.append(PgfDataTable.drawPgfPlotManyCurves("Fit", "axis", 0, 1, pgfTblList, 2,
					"", " "));
			sb.append(PgfDataTable.pgfplotEndDocument());
			fw = new FileWriter(filepath + "smoothness" + rhoName + "/" + coeffName + ".tex");
			fw.write(sb.toString());
			fw.close();

			sb.setLength(0);
			sb.append(fit.evaluate(X[0]));
			for (int s = 1; s < X.length; s++)
				sb.append("," + fit.evaluate(X[s]));
			fw = new FileWriter(filepath + "smoothness" + rhoName + "/" + coeffName + ".txt");
			fw.write(sb.toString());
			fw.close();
		}
		System.out.println("A -- O K ! ! !");
	}

}
