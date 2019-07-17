package umontreal.ssj.latnetbuilder.weights;

import java.util.ArrayList;
import java.util.List;

/**Many point set constructions for (randomized) quasi-Monte Carlo integration and -simulation resort to search 
 * algorithms for a *good* @ref Rank1Lattice such as those implemented in *LatNetBuilder*, for instance.
 * These algorithms commonly make use of certain *figures of merit* to assess the quality of the @ref Rank1Lattice.
 * In order to be able to obtain more influence on these figures of merit, e.g. to amend for a high
 * variability of one projection of the integrand, one commonly may pass @ref Weights to the search algorithm. 
 * Well-known types are @ref ProductWeights as well as @ref ProjectionDependentWeights and their subclasses
 *  @ref OrderDependentWeights, @ref GeometricWeights.
 * 
 * 
 * This class implements such weights. The template parameter 'T' indicates the data type of the underlying index. 
 * Common choices are integers (e.g. the order for @ref OrderDependentWeights, @ref CoordinateSet for @ref ProjectionDependentWeights, etc.).
 * @author florian
 *
 *@param <T> indicates the data type of the index of the weight (e.g. @ref CoordinateSet, @ref Integer,...).
 */
public abstract class Weights<T>{
	
	
	protected double defaultWeight = 1.0; //weight to be used for indices that were not explicitly set.
	protected ArrayList<SingletonWeight<T>> weights; //actual weights
//	protected double weightPower = 2.0;
	
	/**
	 * Constructs weights from a list of @ref SingletonWeights.
	 * @param w list of weights with index type 'T'.
	 */
	public Weights(List<SingletonWeight<T>> w) {
		weights = new ArrayList<SingletonWeight<T>>(w);
	}
	
	
	/**
	 * Initializes an empty list of weights.
	 */
	public Weights() {
		weights = new ArrayList<SingletonWeight<T>>();
	}

	/**
	 * Returns the current weights.
	 * @return the list of weights.
	 */
	public ArrayList<SingletonWeight<T>> getWeights(){
		return weights;
	}
	
	/**
	 * Sets 'dWeight' as the current 'defaultWeight'.
	 * @param dWeight desired default weight.
	 */
	public void setDefaultWeight(double dWeight) {
		defaultWeight = dWeight;
	}
		
	/**
	 * Returns the current 'defaultWeight'.
	 * @return the value of the default weight.
	 */
	public double getDefaultWeight() {
		return defaultWeight;
	}
	
//	public void setWeightPower(double p){
//		weightPower = p;
//	}
	
//	public double getWeightPower(){
//		return weightPower;
//	}
	
	/**
	 * Adds a new weight to the list. In case the weight for the respective index had already been set, it is 
	 * overwritten.
	 * @param singletonWeight weight to be added.
	 */
	//adds a new weight. in case the weight for the respective index had already been set, it is overwritten.
	public void add(SingletonWeight<T> singletonWeight) {
		boolean added = false;
		//TODO: find early exit and remove it from if(...)
		for(SingletonWeight<T> w : weights ) {
			if(w.getIndex() == singletonWeight.getIndex() && (!added)) {
				weights.set(weights.indexOf(w),singletonWeight);
				added = true;
			}
		}
		if(!added)
			weights.add(singletonWeight);
	}
	
	/**
	 * Adds a new weight with index 'index' and weight 'weight' or overwrites it, if the index already exists in
	 * the list.
	 * @param index  index of the weight to be added.
	 * @param weight  value of the weight to be added.
	 */
	public void add(T index, double weight) {
		add(new SingletonWeight<T>(index,weight));
	}
	
	
	
	/**
	 * Basic formatted string-output.
	 */
	public String toString() {
		StringBuffer sb = new StringBuffer("");
		sb.append("Weights [default = " + getDefaultWeight() + "]\n");
		
		sb.append("[");
		for(SingletonWeight<T> w : weights)
			sb.append(w.getWeight() + ",");
		sb.deleteCharAt(sb.length()-1);
		sb.append("]\n");
		return sb.toString();
	}	
	
	/*
	 * Methods to be implemented for non-abstract class
	 */
	/**
	 * Provides a String that can be interpreted by the command line interface of LatNetBuilder.
	 * @return String for LatNetBuilder.
	 */
	public abstract String toLatNetBuilder();


	
}
