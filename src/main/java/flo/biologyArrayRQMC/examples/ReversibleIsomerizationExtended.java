package flo.biologyArrayRQMC.examples;

import umontreal.ssj.rng.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import flo.neuralNet.NeuralNet;
import umontreal.ssj.hups.LMScrambleShift;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.hups.SobolSequence;
import umontreal.ssj.markovchainrqmc.*;
import umontreal.ssj.util.Chrono;
import umontreal.ssj.util.PrintfFormat;
import umontreal.ssj.util.sort.MultiDim;
import umontreal.ssj.util.sort.MultiDim01;
import umontreal.ssj.probdist.DiscreteDistribution;
import umontreal.ssj.probdist.DiscreteDistributionInt;
import umontreal.ssj.probdist.NormalDist;
import umontreal.ssj.probdist.PoissonDist;
import umontreal.ssj.stat.TallyStore;

//class ReversibleisomerisationComparable extends MarkovChainComparable   implements MultiDim01{
public class ReversibleIsomerizationExtended extends ChemicalReactionNetwork implements MultiDim01 {

	double N0; //Total number of molecules

	public ReversibleIsomerizationExtended(double[] c, double[] X0, double tau, double T, double N0) {
		this.c = c;
		this.X0 = X0;
		this.tau = tau;
		this.T = T;
//		S = new double[][] { { -1, 1 ,0,0}, { 1, -1,0,0 }, {0,0,1,-1}  };
		S = new double[][] { { -1, 1 ,0,0}, { 1, -1,0 ,0},{0,0,1,-1},{0,0,1,-1},{0,0,1,-1} ,{0,0,1,-1},{0,0,1,-1},{0,0,1,-1},{0,0,1,-1} }; //more species --> more lines
//		S = new double[][] { { -1, 1 ,0,0, 0,0, 0,0,  0,0}, { 1, -1 ,0,0 ,0,0, 0,0,  0,0},{0,0,1,-1, 1,-1, 1,-1,  1,-1} };  //More reactions -> more columns

		init();
		this.N0 = N0;
	}

	public double getPerformance() {
		return X[0];

	}

	public double[] getState() {
		return X;
	}

	public int compareTo(MarkovChainComparable m, int i) {
		if (!(m instanceof ReversibleIsomerizationExtended)) {
			throw new IllegalArgumentException(
					"Can't compare an ReversibleIsomerizationExtended with other types of Markov chains.");
		}
		double mx;

		mx = ((ReversibleIsomerizationExtended) m).X[i];
		return (X[i] > mx ? 1 : (X[i] < mx ? -1 : 0));

	}

	public String toString() {
		StringBuffer sb = new StringBuffer("----------------------------------------------\n");
		sb.append(" ReversibleIsomerisation:\n");
		sb.append("X0 =\t" + "{" + X0[0] + ", " + (N0-X0[0]) + ", " + X0[1] + "}\n");
		sb.append("N =\t" + (N) + "\n");
		sb.append("K =\t" + K + "\n");
		sb.append("c =\t" + "{" + c[0] + ", " + c[1] + ", " + c[2] +"}\n");
		sb.append("T =\t" + T + "\n");
		sb.append("tau =\t" + tau + "\n");
		sb.append("steps =\t" + numSteps + "\n");
		sb.append("----------------------------------------------\n\n");

		return sb.toString();
	}

	@Override
	public void computePropensities() {
		double x1 = (N0-X[0]);
		a[0] = c[0] * X[0];
		a[1] = c[1] * x1;
//		a[2] = c[2] * X[1] ;
//		a[3] = c[3] * X[1];
//		a[4] = c[4] * X[1];
//		a[5] = c[5] * X[1];
//		a[6] = c[6] * X[1];
//		a[7] = c[7] * X[1];
//		a[8] = c[8] * X[1];
//		a[9] = c[9] * X[1];
		a[2] = c[2] * X[1] * X[2]  * X[3]* X[4]  * X[5] * X[6] * X[7]; //more species
		a[3] = c[3] * X[1] * X[2] * X[3] * X[4]   * X[5] * X[6] * X[7];
		
	}
	
	

	@Override
	public double[] getPoint() {
		double[] state01 = new double[N];
    	for(int i=0;i<N;i++)
        state01[i] = getCoordinate(i);       
        return state01;
	}

	@Override
	public double getCoordinate(int j) {
		double zvalue;                 
		double x01 = N0 - X0[0];

    	
    	switch (j) {
        case 0:   
        	zvalue = (X[j]- X0[j] - (-c[0]*X0[0]+c[1]*x01)*step*tau )/(Math.sqrt((c[0]*X0[0]+c[1]*x01)*step*tau));
            
        	return NormalDist.cdf01 (zvalue);
//        case 1:   //more reactions
//        	zvalue = (X[j]- X0[j] - tau * step * X0[1] *( c[2] - c[3] + c[4] - c[5]+ c[6]-c[7]  +c[8]-c[9]))/(tau*step*Math.sqrt( X0[1]*(c[2]+ c[3] + c[4] + c[5] + c[6] + c[7] + c[8] + c[9])));
//        	return NormalDist.cdf01 (zvalue);
        case 1:   //more species
        	zvalue = (X[j]- X0[j] - tau * step * X0[1] * X0[2]*X0[3]*X0[4]*X0[5]*X0[6]*X0[7]*( c[2] - c[3]))/(tau*step*Math.sqrt( X0[1]* X0[2]*X0[3]*X0[4] *X0[5]* X0[6] * X0[7]*(c[2]+ c[3])));
        	return NormalDist.cdf01 (zvalue);
        	
        case 2:   
        	zvalue = (X[j]- X0[j] - tau * step * X0[1] * X0[2]*X0[3]*X0[4]*X0[5]*X0[6]*X0[7]*( c[2] - c[3]))/(tau*step*Math.sqrt( X0[1]* X0[2]*X0[3]*X0[4] *X0[5]* X0[6] * X0[7]*(c[2]+ c[3])));
        	return NormalDist.cdf01 (zvalue);
        case 3:   
        	zvalue = (X[j]- X0[j] - tau * step * X0[1] * X0[2]*X0[3]*X0[4]*X0[5]*X0[6]*X0[7]*( c[2] - c[3]))/(tau*step*Math.sqrt( X0[1]* X0[2]*X0[3]*X0[4] *X0[5]* X0[6] * X0[7]*(c[2]+ c[3])));
        	return NormalDist.cdf01 (zvalue);
        case 4:   
        	zvalue = (X[j]- X0[j] - tau * step * X0[1] * X0[2]*X0[3]*X0[4]*X0[5]*X0[6]*X0[7]*( c[2] - c[3]))/(tau*step*Math.sqrt( X0[1]* X0[2]*X0[3]*X0[4] *X0[5]* X0[6] * X0[7]*(c[2]+ c[3])));
        	return NormalDist.cdf01 (zvalue);
        case 5:   
        	zvalue = (X[j]- X0[j] - tau * step * X0[1] * X0[2]*X0[3]*X0[4]*X0[5]*X0[6]*X0[7]*( c[2] - c[3]))/(tau*step*Math.sqrt( X0[1]* X0[2]*X0[3]*X0[4] *X0[5]* X0[6] * X0[7]*(c[2]+ c[3])));
        	return NormalDist.cdf01 (zvalue);
        case 6:   
        	zvalue = (X[j]- X0[j] - tau * step * X0[1] * X0[2]*X0[3]*X0[4]*X0[5]*X0[6]*X0[7]*( c[2] - c[3]))/(tau*step*Math.sqrt( X0[1]* X0[2]*X0[3]*X0[4] *X0[5]* X0[6] * X0[7]*(c[2]+ c[3])));
        	return NormalDist.cdf01 (zvalue);
        case 7:   
        	zvalue = (X[j]- X0[j] - tau * step * X0[1] * X0[2]*X0[3]*X0[4]*X0[5]*X0[6]*X0[7]*( c[2] - c[3]))/(tau*step*Math.sqrt( X0[1]* X0[2]*X0[3]*X0[4] *X0[5]* X0[6] * X0[7]*(c[2]+ c[3])));
        	return NormalDist.cdf01 (zvalue);
        default:
            throw new IllegalArgumentException("Invalid state index");
    	}
	}
	
	

}
