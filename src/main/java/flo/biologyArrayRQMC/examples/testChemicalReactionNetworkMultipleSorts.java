package flo.biologyArrayRQMC.examples;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.hups.SortedAndCutPointSet;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChains;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChainsMultipleSorts;
import umontreal.ssj.markovchainrqmc.ArrayOfComparableChainsNN;
import umontreal.ssj.markovchainrqmc.MarkovChainComparable;
import umontreal.ssj.rng.MRG32k3a;
import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.SplitSort;

public class testChemicalReactionNetworkMultipleSorts {

	public static void main(String[] args) throws IOException {

		ChemicalReactionNetwork model;
		ArrayList<MultiDimSort> sortList = new ArrayList<MultiDimSort>();

		double[]c = {8.696E-5, 0.02, 1.154E-4,0.02,0.016,0.0017};//Nano: 1E-9
		double[] x0 = {33000.0,33030.0, 1100.0, 1100.0, 1100.0, 1100.0};
		double T = 0.00005;
		double tau = T/20.0;
		int numSteps = (int) (T/tau);

		
		
		 model = new PKA(c,x0,tau,T);
		 System.out.println(model.toString());
		 String modelDescription = "PKALessNoise";
		 String dataLabel = "MCDataLessNoise";
			String sortLabel = "conditional";
			
//		for(int i = 0; i < numSteps; i++)
//			sortList.add(new PKASortConditionalLinear(c,model.N,T-(double)i * tau));
			
/* NORMAL			
		double[]	a = {20028.18004554095, 20028.18004554095, 19793.36287562981, 
					   19334.138241310644, 18766.12275110118, 18217.722949586576, 
					   17690.467771441403, 17195.096273117982, 16739.962883652475, 
					   16316.474236332935, 15929.354419567386, 15563.46652499161, 
					   15220.122122708604, 14895.850463616702, 14586.047232214736, 
					   14294.319532873596, 14008.868866712313, 13736.59183000939, 
					   13473.938920590428, 13217.61974726176};
		double[]		b = {0.03427679200199507, 0.03427679200199507, 0.06539508374216384, 
					   0.09634448283766185, 0.12615877846880408, 0.15304455817712634, 
					   0.17791706890835096, 0.2008405236111551, 0.22172837799613024, 
					   0.2408547687081658, 0.2584476114277034, 0.27502958066390565, 
					   0.2903928146367952, 0.3050295559451786, 0.31878883094309146, 
					   0.3317725616308061, 0.34443443938659246, 0.3565639443673749, 
					   0.3682094580639178, 0.37958978333571436};
		double[]		cc = {0.007126352584318434, 0.007126352584318434, 
					   0.0014374981258985123, 0.0005348764144755397, 
					   0.0024027532937657852, 0.005706451752545136, 0.010776841846678915, 
					   0.01748770007188198, 0.02548754674573592, 0.034905985807408405, 
					   0.045058526890338645, 0.056534827119851010, 0.06913411114669063, 
					   0.08276119611572215, 0.09769271962855322, 0.11350505163447691, 
					   0.1306738860702969, 0.14866424533956718, 0.16780389241741026, 
					   0.18805260034326338};
		double[]		d = {-0.05139705135474862, -0.05139705135474862, 
					-0.10989177565381278, -0.14049943751870497, -0.15711648053663607, 
					-0.1670269979629234, -0.1735634244448111, -0.17792682645378544, 
					-0.1811436637816329, -0.18337905279752004, -0.18527130556112376, 
					-0.18672102447390226, -0.1877667332062381, -0.18861786467953084, 
					-0.1892425930322338, -0.1897375362181242, -0.1901444331345016, 
					-0.19048896250617628, -0.19067210358403267, -0.19082463522178256};
		double[]			e = {0.4271818099269067, 0.4271818099269067, 0.3809944594981382, 
					   0.3575074570285563, 0.34482291187345815, 0.33801008764651314, 
					   0.3315924164089207, 0.32459907654688974, 0.31755867441245156, 
					   0.31040176973716527, 0.3037803499203231, 0.2967831072051265, 
					   0.2895625985685569, 0.2822849711517792, 0.27475406532067664, 
					   0.26723906439509515, 0.25952390322658137, 0.2518236793394717, 
					   0.24382214607159747, 0.23580494104827196};
		double[]			f = {0.22161374718122634, 0.22161374718122634, 0.191638146913436, 
					   0.17431383479847848, 0.1707729217158348, 0.16725167273690053, 
					   0.16370957660016305, 0.15925074131415234, 0.15534941885406753, 
					   0.1523310074684232, 0.14905199773748462, 0.14479993345930164, 
					   0.14239552770993205, 0.13793713047464468, 0.13534307019104524, 
					   0.13247267318500838, 0.1291074418795342, 0.1255050648102007, 
					   0.12209941683770008, 0.11810805274449772};
		double[]		g = {0.11080687359061317, 0.11080687359061317, 0.095819073456718, 
					   0.08715691739923924, 0.0853864608579174, 0.08362583636845027, 
					   0.08185478830008153, 0.07962537065707617, 0.07767470942703376, 
					   0.0761655037342116, 0.07452599886874231, 0.07239996672965082, 
					   0.07119776385496603, 0.06896856523732234, 0.06767153509552262, 
					   0.06623633659250419, 0.0645537209397671, 0.06275253240510036, 
					   0.06104970841885004, 0.05905402637224886};
*/
			//reduced noise
		double[]	a = {3850.610985166324, 3850.610985166324, 1753.0250620879997, 
				   2329.5365894892966, 2413.7161695387294, 2047.4243729638838, 
				   1721.3642132714413, 1767.981196158038, 1568.6347255989604, 
				   1459.78503355266, 1357.5737884481787, 1143.5623431618988, 
				   927.9610657238554, 841.1217438035601, 834.0959506911531, 
				   695.2946908287777, 598.4388240666194, 450.20222495000417, 
				   322.7631908007861, 206.1966095762691};
		double[]		b = {0.8961202040145658, 0.8961202040145658, 0.9090619073333861, 
				   0.917272608755699, 0.9268328482824759, 0.9352072473604245, 
				   0.9428005703234975, 0.9485517860153747, 0.9543974860844114, 
				   0.9592364932854699, 0.9630993382419972, 0.967517455504309, 
				   0.9718735846641853, 0.9758380924680672, 0.9801820594978927, 
				   0.9837453965461658, 0.9867328316083195, 0.9901239487405835, 
				   0.9929118722417815, 0.9966414063224986};
		double[]	cc = {-0.31034096329489164, -0.31034096329489164, -0.3039062747348334, 
				-0.29078720240150885, -0.28316770634622873, -0.273841698694772, 
				-0.26288478431351275, -0.2504200644379609, -0.23695419974593132, 
				-0.22282458387743106, -0.20713792133768094, -0.19080600085801866, 
				-0.17250387798324104, -0.15439079744497702, -0.1354197289155347, 
				-0.11601815916416479, -0.09520167231885349, -0.07287528715337446, 
				-0.05009080760349783, -0.02522406462405831};
		double[]	d = {0.1751944115020413, 0.1751944115020413, 0.1258905517645592, 
				   0.1001049955279909, 0.07093223761036244, 0.0527438990606504, 
				   0.041977022049477807, 0.032868422674355934, 0.024034436857006677, 
				   0.02096182040366733, 0.01667843073490174, 0.013935500399306248, 
				   0.010424837919228531, 0.007387876618144721, 0.004102135538890376, 
				   0.002459185757528692, 0.001150829662014894, 0.0007467053656843599, 
				   0.0007404115961722618, 0.0004189966548389335};
		double[]	e = {0.03493020275513948, 0.03493020275513948, -0.00350282532327356, 
				   0.021578266522875283, 0.005060250652976301, 0.004721124581262313, 
				   0.0032651727968755545, 
				   0.0044529624633808645, -0.0041676410944242485, 
				   0.002223546812046142, 0.006488175973028641, 0.007916980762877791, 
				   0.004271610253133103, 
				   0.002179078033814742, -0.001040766524620232, 
				-0.0030160107001859907, -0.0031703230003414977, 
				-0.0018430799556573318, 0.00280924673314491, 0.001314353781028461};
		double[]	f = {-2.3904190448531377, -2.3904190448531377, 
				   1.364208750273254, -0.27300278602479827, -0.4469515074938772, 
				-0.004041799059760044, 0.3600232040239237, -0.022837381319502133, 
				   0.16292580735137893, 0.022622741781479265, -0.04746343423526332, 
				   0.05890111919304499, 0.2346589910115611, 
				   0.16772953034515045, -0.07117126930713669, -0.04402194563466205, 
				-0.08104179494817798, -0.08151301478184612, -0.10660174907245189, 
				-0.15111412538329735};
		double[]		g = {1.2716782519727854, 1.2716782519727854, -0.6785949027312222, 
				   0.18882451297850397, 0.24260516869148233, 
				   0.015413047050299111, -0.20303574069351504, 
				   0.009978534647590766, -0.09469032313790987, -0.015148040958719678, 
				   0.02063692429188079, -0.029582492589894885, -0.12990186423229944, 
				-0.09461087563317855, 0.04127861078943428, 0.03247422589780763, 
				   0.04982991927195494, 0.054403052259946016, 0.0640311217382127, 
				   0.08371051637407069};
				
		for (int i = 0; i < a.length; i++) {
			sortList.add(new PKASort(a[i],b[i],cc[i],d[i],e[i],f[i],g[i]));
		}
		
/*		double[] c = { 3E-7, 1E-4, 1E-3, 3.5 };
		double[] x0 = { 250.0, 1E5 };
		double N0 = 2E5 + 250.0 + 1E5;
		double T = 4;
		double tau = 0.2;

		model = new SchloeglSystemProjected(c, x0, tau, T,N0);
		String modelDescription = "SchloeglSystemProj";
		String dataLabel = "MCData";
		String sortLabel = "l1";

		ArrayList<MultiDimSort> sortList = new ArrayList<MultiDimSort>();
	double[]	a = {258.08424900431476, 258.08424900431476, 261.48079476115794, 
				   263.4428886265869, 264.8616180800489, 266.305793668144, 
				   267.53304006024337, 268.6651710041849, 269.69495593902553, 
				   270.53027085596557, 271.3720774552504, 272.102275628243, 
				   272.59645204814956, 273.06747203948294, 273.5221092642125, 
				   273.9659256193167, 274.40425544095166, 274.9585940584114, 
				   275.57259995678515, 276.6022004917788};
	double[]		b = {-183.80616601113414, -183.80616601113414, -182.607100941182, 
				-183.0113463601882, -182.60033595073241, -182.74224034886402, 
				-183.03298944270117, -183.3026018888793, -183.7340529921913, 
				-184.47510841363902, -185.3685660345213, -186.5548323910866, 
				-188.20757076613, -190.6684160043247, -194.10142830340257, 
				-199.6931961856299, -208.5797950130828, -223.3778391917219, 
				-251.23969976999376, -320.57268446165193 };
	double[]		cc = {-0.020718048328757437, -0.020718048328757437, 
				-0.02085665892524057, -0.02050430972354262, -0.020321468483031433, 
				-0.01994455249078135, -0.0193665061750056, -0.01889693750412154, 
				-0.018155375927028216, -0.01719047333196788, -0.016191738209059798, 
				-0.015095729923674129, -0.01383336327857643, -0.012552046103108346, 
				-0.011278578982519468, -0.009914829637951084, -0.00850287287400984, 
				-0.007041211544763277, -0.0054769292061915996, -0.0037027507157621027};
	double[]	d = {-0.0004121763550424434, -0.0004121763550424434, 
				-0.0003172872596619393, -0.0002984301587111952, 
				-0.00030416451356721127, -0.0002894964829811772, 
				-0.00026188343798948103, -0.00028320414914643414, 
				-0.0002562027307717224, -0.00022520036419196947, 
				-0.00020498750823635462, -0.00018941651036391923, 
				-0.00016077589696169295, -0.0001475501369799153, 
				-0.0001330120153858751, -0.00011591745302139875, 
				-0.00009408579516639558, -0.00006673785744387753, 
				-0.00003785054779129977, -0.000013611715811696467};
	double[]	e = {46.44815806307885, 46.44815806307885, 37.00570410155215, 
				   35.02706326387889, 35.537353612864536, 33.969420886161004, 
				   31.066626075015538, 33.04259381385883, 30.161600693699768, 
				   26.828897216859286, 24.556321027528586, 22.717335558885758, 
				   19.547647654439107, 17.890555286673614, 16.108003871244392, 
				   14.051956607085222, 11.522227231467363, 8.444783296776354, 
				   5.195040493861866, 2.3474355837984646};
		for(int i = 0; i < a.length; i++) {
//			sortList.add(new SchloeglSystemProjectedSort(a[i],b[i],cc[i],d[i],e[i]));
			sortList.add(new ProjectedL1Sort(model.X0,c,(double)i,N0));
		}
*/
		ArrayOfComparableChainsMultipleSorts chain = new ArrayOfComparableChainsMultipleSorts(model,sortList);

//		 int[] N = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
//		 524288, 1048576 }; // n from 8
		// to 20.
		int[] N = {262144,
				 524288, 1048576}; // n from 8
		

		int[] logN = { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
		int mink = 18;
		int numSets = N.length;

		int sortCoordPtsList = 1;
	
		
		// MultiDimSort sortPointSet = new SchloeglSystemSort(); //set here if
		// sortCoordPts>1

		int m = 50;

		StringBuffer sb = new StringBuffer("");
		String str;
		String outFile = modelDescription + ".txt";

		RandomStream stream = new MRG32k3a();
		RQMCPointSet[] rqmcPts;
		PointSet[] pointSets = new PointSet[numSets];
		PointSetRandomization rand;
		RQMCPointSet prqmc;
		int s;

		int nMC = (int) 1E6; // n to estimate MC variance.
		Tally statMC = new Tally();
		statMC.init();
		// model.simulRunsWithSubstreams(nMC, model.numSteps, stream, statMC);
		model.simulRuns(nMC, model.numSteps, stream, statMC);
		double varMC = statMC.variance();
		str = "\n\n --------------------------\n";
		str += "MC average  = " + statMC.average() + "\n";
		str += "MC variance = " + varMC + "\n\n";
		sb.append(str);
		System.out.println(str);

		
		
			
			ArrayList<RQMCPointSet[]> listP = new ArrayList<RQMCPointSet[]>();

			// Independent points (Monte Carlo)
			// rqmcPts = new RQMCPointSet[numSets];
			// for (s = 0; s < numSets; ++s) {
			// pointSets[s] = new IndependentPointsCached(N[s], model.K + model.N);
			// rand = new RandomShift(stream);
			// prqmc = new RQMCPointSet(pointSets[s], rand);
			// rqmcPts[s] = prqmc;
			// }
			// rqmcPts[0].setLabel("Independent points");
			// listP.add(rqmcPts);

			// Stratification
			// rqmcPts = new RQMCPointSet[numSets];
			// int k;
			// for (s = 0; s < numSets; ++s) {
			// k = (int) Math.round(Math.pow(Num.TWOEXP[s + mink], 1.0 / (double)
			// (sortCoordPts + model.K)));
			// pointSets[s] = new StratifiedUnitCube(k, sortCoordPts + model.K);
			// // Here the points must be sorted at each step, always.
			// // In the case of Hilbert map, the points should be 2d and sorted
			// // based on one coordinate,
			// // whereas the states are 2d and sorted by the Hilbert sort.
			// rand = new RandomShift(stream);
			// prqmc = new RQMCPointSet(pointSets[s], rand);
			// rqmcPts[s] = prqmc;
			// }
			// rqmcPts[0].setLabel("Stratification");
			// listP.add(rqmcPts);

			// Sobol + LMS
			rqmcPts = new RQMCPointSet[numSets];
			for (s = 0; s < numSets; ++s) {
				if (sortCoordPtsList == 1)
					pointSets[s] = new SobolSequence(s + mink, 31, 1 + model.K);
				else {

					pointSets[s] = new SortedAndCutPointSet(
							new SobolSequence(s + mink, 31, 1 + model.K), sortList.get(19));
				}
				rand = new LMScrambleShift(stream);
				prqmc = new RQMCPointSet(pointSets[s], rand);
				rqmcPts[s] = prqmc;
			}
			rqmcPts[0].setLabel("Sobol+LMS");
			listP.add(rqmcPts);

			// Sobol + LMS + Baker
			// rqmcPts = new RQMCPointSet[numSets];
			// for (s = 0; s < numSets; ++s) {
			// if (sortCoordPts == 1)
			// pointSets[s] = new BakerTransformedPointSet(new SobolSequence(s + mink, 31, 1
			// + model.K));
			// else
			// pointSets[s] = new SortedAndCutPointSet(
			// new BakerTransformedPointSet(new SobolSequence(s + mink, 31, sortCoordPts +
			// model.K)),
			// sortPointSet);
			// // pointSets[s] = new SortedAndCutPointSet(new SobolSequence(s +
			// // mink, 31, sortCoordPts + baseChain.K), sort);
			// rand = new LMScrambleShift(stream);
			// prqmc = new RQMCPointSet(pointSets[s], rand);
			// rqmcPts[s] = prqmc;
			// }
			// rqmcPts[0].setLabel("Sobol+LMS+baker");
			// listP.add(rqmcPts);

			// Sobol+NUS
			// rqmcPts = new RQMCPointSet[numSets];
			// for (s = 0; s < numSets; ++s) {
			// if (sortCoordPts == 1) {
			// CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31, 1 +
			// model.K));
			// p.setRandomizeParent(false);
			// pointSets[s] = p;
			// } else {
			// CachedPointSet p = new CachedPointSet(new SobolSequence(s + mink, 31,
			// sortCoordPts + model.K));
			// p.setRandomizeParent(false);
			// // The points are sorted here, but only once.
			// pointSets[s] = new SortedAndCutPointSet(p, sortPointSet);
			// }
			// rand = new NestedUniformScrambling(stream);
			// prqmc = new RQMCPointSet(pointSets[s], rand);
			// rqmcPts[s] = prqmc;
			// }
			// rqmcPts[0].setLabel("Sobol+NUS");
			// listP.add(rqmcPts);

			for (RQMCPointSet[] ptSeries : listP) {
				String label = ptSeries[0].getLabel();
				str = label;
				str += "\n-----------------------------\n";
				sb.append(str + "\n");
				System.out.println(str);
				// If Stratification, then we need to sort point set in every step
				int sortedCoords =  0;
				str = (chain.testVarianceRateFormat(ptSeries, sortedCoords, model.numSteps, m, varMC,
						modelDescription+ "-" +sortLabel+ "-" + label, label));
				System.out.println(str);
				sb.append(str + "\n");

			}
			
		FileWriter file = new FileWriter(outFile);
		file.write(sb.toString());
		file.close();

	}

}
