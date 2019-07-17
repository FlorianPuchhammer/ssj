package umontreal.ssj.latnetbuilder.weights;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Implements comparable @ref Weights.
 * @author florian
 *
 * @param <T> indicates the data type of the index of the weight (e.g @ref Integer). Needs to be comparable.
 */

//TODO: do not know enough about inheritance to decide if re-listing all methods from the weight-class, only with singletonWeightComparable is 
//		really necessary. 
public abstract class WeightsComparable<T extends Comparable<T>> extends Weights<T> {

	
	protected ArrayList<SingletonWeightComparable<T>> weights; //actual weights
	
	/**
	 * Constructs weights from a list of @ref SingletonWeightsComparable.
	 * @param w list of weights with index type 'T'.
	 */
	public WeightsComparable(List<SingletonWeightComparable<T>> w) {
		weights = new ArrayList<SingletonWeightComparable<T>>(w);
	}
	
	/**
	 * Initializes an empty list of comparable weights.
	 */
	public WeightsComparable() {
		weights = new ArrayList<SingletonWeightComparable<T>>();
	}
	
	/**
	 * Returns the current weights.
	 * @return the list of weights.
	 */
	//TODO: changing return type with getWeights() does not work. do we really need this?
	public ArrayList<SingletonWeightComparable<T>> getComparableWeights(){
		return weights;
	}
	

	
	/**
	 * Adds a new comparable weight to the list. In case the weight for the respective index had already been set, it is 
	 * overwritten.
	 * @param singletonWeight comparable weight to be added.
	 */
	public void add(SingletonWeightComparable<T> singletonWeight) {
		boolean added = false;
		//TODO: find early exit and remove it from if(...)
		for(SingletonWeightComparable<T> w : weights ) {
			if(w.getIndex() == singletonWeight.getIndex() && (!added)) {
				weights.set(weights.indexOf(w),singletonWeight);
				added = true;
			}
		}
		if(!added)
			weights.add(singletonWeight);
	}

	
	

	/**
	 * Sorts the weights w.r.t. the ordering defined on 'T', i.e., on the indices.
	 */
	public void sort() {
		Collections.sort(weights);
	}
	
	/**
	 * Adds a new weight with index 'index' and weight 'weight' or overwrites it, if the index already exists in
	 * the list.
	 * @param index  index of the weight to be added.
	 * @param weight  value of the weight to be added.
	 */
	public void add(T index, double weight) {
		add(new SingletonWeightComparable<T>(index,weight));
	}
	
//	/**
//	 * Basic formatted string-output.
//	 */
//	public String toString() {
//		StringBuffer sb = new StringBuffer("");
//		sb.append("Weights [default = " + getDefaultWeight() + "]\n");
//		
//		sb.append("[");
//		for(SingletonWeightComparable<T> w : weights)
//			sb.append(w.getWeight() + ",");
//		sb.deleteCharAt(sb.length()-1);
//		sb.append("]\n");
//		return sb.toString();
//	}	
	
}
