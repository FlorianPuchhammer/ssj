package flo.optionsArrayRQMC;

import umontreal.ssj.util.sort.MultDimToOneDimSort;

public class VGLinearSort  extends MultDimToOneDimSort{

		double a;
		double b;
		
		
		public VGLinearSort(double a, double b) {
			this.dimension = 2; //This is needed to be 2 for the superclass!
			this.a = a;
			this.b = b;
		}
		
		
		
		@Override
		public double scoreFunction(double[] v) {
//			System.out.println("a: " + a + "\tb: " + b);
//			System.out.println(v[0] + ", " + v[1]);
			return a * v[0] + b* v[1];
		}
		
		@Override
		public String toString() {
			return "VGSortLinear";
		}
}
