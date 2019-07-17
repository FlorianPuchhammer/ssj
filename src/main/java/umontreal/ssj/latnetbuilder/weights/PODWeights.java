package umontreal.ssj.latnetbuilder.weights;

import java.io.IOException;
import java.util.List;

import umontreal.ssj.mcqmctools.anova.CoordinateSet;
import umontreal.ssj.mcqmctools.anova.CoordinateSetLong;

public class PODWeights extends Weights{
	
	ProductWeights productWeights;
	OrderDependentWeights orderDependentWeights;
	
	public PODWeights(ProductWeights pWeights,OrderDependentWeights odWeights) {
		this.productWeights = pWeights;
		this.orderDependentWeights = odWeights;
	}
	
	public PODWeights(ProductWeights pWeights) {
		this.productWeights = pWeights;
		this.orderDependentWeights = new OrderDependentWeights();
	}
	
	public PODWeights(OrderDependentWeights odWeights) {
		this.productWeights = new ProductWeights();
		this.orderDependentWeights = odWeights;
	}
	
	public PODWeights() {
		this.productWeights = new ProductWeights();
		this.orderDependentWeights = new OrderDependentWeights();
	}


	
	public ProductWeights getProductWeights() {
		return productWeights;
	}

	public void setProductWeights(ProductWeights productWeights) {
		this.productWeights = productWeights;
	}

	public OrderDependentWeights getOrderDependentWeights() {
		return orderDependentWeights;
	}

	public void setOrderDependentWeights(OrderDependentWeights orderDependentWeights) {
		this.orderDependentWeights = orderDependentWeights;
	}

	public void addOrderDependentWeight(SingletonWeightComparable<Integer> weight) {
		orderDependentWeights.add(weight);
	}
	
	public void addOrderDependentWeight(int ord, double weight) {
		orderDependentWeights.add(ord,weight);
	}
	
	public void addProductWeight(SingletonWeightComparable<Integer> weight) {
		productWeights.add(weight);
	}
	
	public void addProductWeight(int index, double weight) {
		productWeights.add(index,weight);
	}
	
	@Override
	public String toString(){
		StringBuffer sb = new StringBuffer("");
		sb.append("POD weights:\n");
		sb.append(productWeights.toString() + "\n");
		sb.append(orderDependentWeights.toString());
		return sb.toString();
	}

	@Override
	public String toLatNetBuilder() {
		StringBuffer sb = new StringBuffer("");
		sb.append("POD:" + orderDependentWeights.getDefaultWeight());
		if(orderDependentWeights.weights.size() > 0) {
			sb.append(":");
		sb.append(orderDependentWeights.printBody());
		}
		sb.append(":" + productWeights.getDefaultWeight());
		if(productWeights.weights.size() > 0) {
			sb.append(":");
		sb.append(productWeights.printBody());
		}
		return sb.toString();
	}
	
	public static void main(String[] args) throws IOException {
		PODWeights myWeights = new PODWeights();
		OrderDependentWeights orderWeights = new OrderDependentWeights();
		orderWeights.add(4,0.33);
		orderWeights.add(1,0.23);

		
		myWeights.setOrderDependentWeights(orderWeights);
		myWeights.addOrderDependentWeight(2, 0.7);
		myWeights.addOrderDependentWeight(1, 0.37);
		
		ProductWeights pWeights = new ProductWeights();
		pWeights.add(7,0.3);
		pWeights.add(0,0.05);
		
		pWeights.setDefaultWeight(1.2);
		pWeights.add(0,0.999);
		
		myWeights.setProductWeights(pWeights);
		
		System.out.println(myWeights.toString());
		System.out.println(myWeights.toLatNetBuilder());

		
//		myWeights.setFileDir("/home/florian/misc/eclipse/ssj-develop/");
//		//myWeights.setFileDir("/u/puchhamf/misc/workspace/");
//		myWeights.setFileName("order_dependent_test.dat");
//		myWeights.write();
		
		System.out.println(" A -- O K !");
	}
	
	
}
