package umontreal.ssj.latnetbuilder.weights;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Class implementing a geometric version of @ref OrderDependentWeights. In particular, this means that each weight is given by a #baseWeight raised 
 * to the power of the order - 1 of the projection it is assigned to.
 * @author florian
 *
 */

public class GeometricWeights extends OrderDependentWeights {

	protected double baseWeight = 1.0;
	protected int truncationLevel;
	protected double c;
	/**
	 * Constructs a geometric weight with given #baseWeight 'base' and #truncationLevel 'truncation'.
	 * @param base desired base value to compute the weights.
	 * @param truncation order up to which geometric weights are set. Higher-order projections will be assigned the #defaultWeight.
	 */
	
	public GeometricWeights(double base, int truncation,double c) {
		super();
		baseWeight = base;
		truncationLevel = truncation;
		this.c = c;
		setWeights();
	}
	
	public GeometricWeights(double base, int truncation) {
		this(base,truncation,1.0);
	}
	
	public double getC() {
		return c;
	}

	public void setC(double c) {
		this.c = c;
	}

	public GeometricWeights(){
		super();
		truncationLevel = 0;
	}
	/**
	 * Returns the #baseWeight.
	 * @return the base value to compute the weights.
	 */
	public double getBaseWeight() {
		return baseWeight;
	}
	
	/**
	 * Sets the #baseWeight.
	 * @param the desired base value to compute the weights.
	 */
	public void setBaseWeight(double base) {
		baseWeight = base;
	}
	
	/**
	 * Returns the current #truncationLevel. Higher-order projections will be assigned the #defaultWeight.
	 * @return truncation order up to which the weights are computed.
	 */
	public int getTruncationLevel() {
		return truncationLevel;
	}
	
	/**
	 * Sets the #truncationLevel. Higher-order projections will be assigned the #defaultWeight.
	 * @param trLevel desired order up to which the weights are computed.
	 */
	public void setTruncationLevel(int trLevel) {
		truncationLevel = trLevel;
	}

	//TODO: does adding weights make sense? if not, keep it unhandled or like below or with exception?
/*	@Override
	public void add(SingletonWeight<Integer> singeltonWeight){
		System.out.println("WARNING: the method 'add' does nothing for GeometricWeights");
	}
*/
	
	/**
	 * Computes the weights up to the order #truncationLevel and assigns them.
	 */
	public void setWeights() {
		double w = 1.0;
		weights = new ArrayList<SingletonWeightComparable<Integer>>(truncationLevel);
		weights.add(0, new SingletonWeightComparable<Integer>(1,0));
		for(int order = 2; order <= truncationLevel; order++) {
			weights.add(order-1, new SingletonWeightComparable<Integer>(order,c*w));
			w *= baseWeight;
		}
	}
	
	/**
	 * Creates a formatted output of the geometric order dependent weights ordered w.r.t. to the order they are assigned to.
	 * @return a formatted output of the geometric weights.
	 */
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer("");
		sb.append("Geometric order dependent weights [default = " + getDefaultWeight() + ", base = " + baseWeight + "]:\n");
		if(weights.size() > 0)
			sb.append("[");
		sb.append(printBody());
		return sb.toString() + ( weights.size() > 0 ? "]": "");
		}

	
	public static void main(String[] args) throws IOException {
		double base =0.955;
		int trLevel = 5;
		GeometricWeights myWeights = new GeometricWeights(base,trLevel);
		//myWeights.setWeights();
		System.out.println(myWeights.toString());
		System.out.println(myWeights.toLatNetBuilder());
		
//		myWeights.setFileDir("/home/florian/misc/eclipse/ssj-develop/");
//		myWeights.setFileName("order_dependent_test.dat");
//		myWeights.write();
		
		System.out.println(" A -- O K !");
	}
	

}
