package umontreal.ssj.latnetbuilder.weights;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

//TODO: many methods are the same as with product weights. should we implement it as a
//	child class, even though logically it is not?
/**
 * Class implementing *order dependent weights.* This means that all projections of the coordinates of the underlying model of the same order are
 * assigned the same weight. For instance, the projection to the coordinates @f$(x_1,x_2,x_3)@f$ is assigned the same weight as every 
 * other projection of order three, such as @f$(x_2,x_4,x_6)@f$. Mind that the order (i.e., the index of the @ref SingletonWeightComparable) is at least
 * 1 for non-empty projections.
 * @author florian
 *
 */
public class OrderDependentWeights extends WeightsComparable<Integer> {

	protected String fileDir = ""; //indicates output directory for file-output.
	protected String fileName = ""; //indicates filename for the file to generated.
	protected boolean sorted = false; //flag, whether weights are sorted.
	
	/**
	 * Constructs an instance of order dependent weights from a given list of comparable weights.
	 * @param weightList list of comparable weights.
	 */
	public OrderDependentWeights(ArrayList<SingletonWeightComparable<Integer>> weightList) {
		super(weightList);
	}
	
	/**
	 * Default constructor.
	 */
	public OrderDependentWeights() {
		super();
	}

	/**
	 * Sets the directory to which an output-file can be generated.
	 * @param dir path to an output directory.
	 */
	public void setFileDir(String dir) {
		 fileDir = dir;
	}
	
	/**
	 * Returns the directory to which an output-file can be generated.
	 * @return path to an output directory.
	 */
	public String getFileDir() {
		return fileDir;
	}
	
	/**
	 * Sets the name of the file to which an output can be generated.
	 * @param name name for output-file.
	 */
	public void setFileName(String name) {
		fileName = name;
	}
	
	/**
	 * Returns the name of the file to which an output can be generated.
	 * @return name for output-file.
	 */
	public String getFileName() {
		return fileName;
	}
	
	/**
	 * Writes a file with name #fileName to the directory #fileDir containing information on the weights. This file is formatted to be further
	 * processed by *LatticeBuilder.*
	 * @throws IOException
	 */
	public void write() throws IOException {
		 FileWriter file = new FileWriter(fileDir + fileName);
		 StringBuffer sb = new StringBuffer("");
		 for(SingletonWeightComparable<Integer> w : weights)
			 sb.append("order " + w.getIndex() + ":\t" + w.getWeight() + "\n");
		 sb.append("default:\t" + defaultWeight);
		 file.write(sb.toString());
		 file.close();
	}
	
	/**
	 *Sorts the weights and creates a rudimentary string containing the order dependent weights separated by commas. Missing weights (i.e. a 
	 *weight for an order which has not been set and lies between two orders, for which the weights are specified) are 
	 * filled with 'defaultWeight'. 
	 * @return a string containing the values of the weights separated by commas. 
	 */
	public String printBody() {
		if(!sorted)
			sort();
		StringBuffer sb = new StringBuffer("");
		if(weights.size()>0) {
			int index = 1;
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
	 * Creates a formatted output of the order dependent weights sorted w.r.t. to the order they are assigned to.
	 * @return a formatted output of the order dependent weights.
	 */
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer("");
		sb.append("Order dependent weights [default = " + getDefaultWeight() + "]:\n");
		if(weights.size()>0)
			sb.append("[");
		sb.append(printBody());
		return sb.toString() + ( weights.size() > 0 ? "]": "");
		}

	/**
	 * Creates a string formatted for passing it to *LatticeBuilder*.
	 * @return a formatted string that can be processed by *LatticeBuilder.*
	 */
	public String toLatNetBuilder() {
		StringBuffer sb = new StringBuffer("");
		sb.append("order-dependent:" + getDefaultWeight());
		if(weights.size() > 0)
			sb.append(":");
		sb.append(printBody());
//		sb.append(" -o " + weightPower + " ");
		
		return sb.toString();
	}
	
	public static void main(String[] args) throws IOException {
		OrderDependentWeights myWeights = new OrderDependentWeights();
		myWeights.add(4,0.33);
		myWeights.add(1,0.23);
		myWeights.add(2,0.7);
		System.out.println(myWeights.toString());
		System.out.println(myWeights.toLatNetBuilder());
		
//		myWeights.setFileDir("/home/florian/misc/eclipse/ssj-develop/");
//		myWeights.setFileName("order_dependent_test.dat");
//		myWeights.write();
		
		System.out.println(" A -- O K !");
	}

}
