package umontreal.ssj.latnetbuilder.weights;

import java.util.ArrayList;

/**
 * Class implementing *product weights.* Each weight is assigned to one coordinate of the underlying model. Thus, the
 * index of each 'SingletonWeightComparable' is given by an integer (starting at 0 for the first coordinate) and corresponds 
 * to one specific coordinate.
 * @author puchhamf
 *
 */

public class ProductWeights extends WeightsComparable<Integer>{
	
	//flag if the weights have already been sorted
	boolean sorted = false;
	
	public ProductWeights(ArrayList<SingletonWeightComparable<Integer>> weightList) {
		super(weightList);
	}

	public ProductWeights() {
		super();
	}
	
	
	@Override
	public void add(SingletonWeightComparable<Integer> singletonWeight) {
		super.add(singletonWeight);
		sorted = false;
	}
	
	/**
	 * Sorts the weights by their indices.
	 */
	@Override
	public void sort() {
		super.sort();
		sorted = true;
	}

	
	/**
	 *Sorts the weights and creates a rudimentary string containing the product weights separated by commas. Missing weights (i.e. a weight for a
	 * coordinates which has not been set and lies between two coordinates, whose weights are specified) are 
	 * filled with 'defaultWeight'. 
	 * @return a string containing the values of the weights separated by commas. 
	 */
	public String printBody(){
		if(!sorted)
			sort();
		StringBuffer sb = new StringBuffer("");
		if(weights.size()>0) {
			int index = 0;
			for(SingletonWeightComparable<Integer> w : weights) {
				while(index < w.getIndex()) {
					sb.append(getDefaultWeight() + ",");
					index++;
				}
				sb.append(w.getWeight()+",");
				index++;
			}
			sb.deleteCharAt(sb.length()-1);	
		}
		return sb.toString();
	}
	
	/**
	 * Creates a formatted output of the product weights ordered w.r.t. to the coordinate they are assigned to.
	 * @return a formatted output of the product weights.
	 */
	@Override 
	public String toString() {
		StringBuffer sb = new StringBuffer("");
		sb.append("Product weights [default = " + getDefaultWeight() + "]:\n");
		if(weights.size() > 0)
			sb.append("[");
		sb.append(printBody());
		return sb.toString() + ( weights.size() > 0 ? "]": "");
	}
	
	/**
	 * Creates a string formatted for passing it to *LatticeBuilder*.
	 * @return a formatted string that can be processed by *LatticeBuilder*.
	 */
	public String toLatNetBuilder() {
		StringBuffer sb = new StringBuffer("");
		sb.append("product:" + getDefaultWeight());
		if(weights.size() > 0) {
			sb.append(":");
		sb.append(printBody());
		}
//		sb.append(" -o " + weightPower + " ");
		
		return sb.toString();
	}
	
	public static void main(String[] args) {
		ProductWeights myWeights = new ProductWeights();
		//set the weight associated to the 5th coordinate to 0.33, etc.
		myWeights.add(4,0.33);
		myWeights.add(1,0.23);
		myWeights.add(2,0.7);
		//overwrites weight assoc. to 5th coordinate with 0.99
		myWeights.add(4,0.99);
		//sorts and prints the weights
		System.out.println(myWeights.toString());
		System.out.println(myWeights.toLatNetBuilder());
		

	}
}