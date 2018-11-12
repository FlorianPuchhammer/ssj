package flo;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Scanner;

import umontreal.ssj.functionfit.LeastSquares;

public class TestStuff{
	
	public static void main(String[] args) throws FileNotFoundException {
		
		Scanner sc =new Scanner( new BufferedReader(new FileReader("data/PKA/MCDataLessNoise_Step_19.csv")));;
		int rows = 262144;
		int cols = 7;
		
		double[][] data = new double[rows][cols];
		
		for (int i=0; i<rows; i++) {
            String[] line = sc.nextLine().trim().split(",");
            for (int j=0; j<cols; j++) {
               data[i][j] = Double.parseDouble(line[j]);
            }
         }
		
		
		System.out.println("Data0:\t" + data[0][0]);
		System.out.println("Data last:\t" + data[rows-1][cols-1]);
		
		double[][] vars = new double[rows][cols-1];
		double[] response = new double[rows];
		
		for(int i = 0; i < rows; i ++) {
			response[i] = data[i][cols-1];
			for (int j = 0; j < cols-1; j++)
				vars[i][j] = data[i][j];
		}
		
		double[] reg = LeastSquares.calcCoefficients(vars, response);
		
		
		String str = "" + reg[0];
		for(int i = 1; i < reg.length; i++) {
			str += " + " + reg[i] + " S_" + i;
		}
		System.out.println(" Regression function:\t" + str);
	}

}

