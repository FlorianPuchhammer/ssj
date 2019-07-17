package umontreal.ssj.latnetbuilder.weights;

/**
 * Implements single instances of weights. The template parameter 'T' indicates the 
 * data type of the associated index, i.e., for instance, 'Integer' for the order of 
 * an @ref OrderDependentWeight. In contrast to its superclass @ref SingletonWeight, 
 * singleton weights from this class can be compared via an ordering defined on 'T'.
 *  
 * @author florian
 *
 * @param <T> indicates the data type of the index of the weight (e.g. @ref Integer,...).
 *  This data type needs to be comparable.
 */

public class SingletonWeightComparable<T extends Comparable<T>> extends SingletonWeight<T> implements Comparable<SingletonWeightComparable<T>>{

	/**
	 * Constructs a single comparable weight with given 'index' and 'weight'.
	 * @param ind the desired index.
	 * @param w the desired value for the weight.
	 */
	SingletonWeightComparable(T ind, double w){
		super(ind,w);
	}
	
	

	/**
	 * method to be able to compare the current weight to another based on the ordering
	 * defined on 'T'. This will enable to sort weights by their index eventually.
	 */
	public int compareTo(SingletonWeightComparable<T> w) {
		return getIndex().compareTo(w.getIndex());
	}
}