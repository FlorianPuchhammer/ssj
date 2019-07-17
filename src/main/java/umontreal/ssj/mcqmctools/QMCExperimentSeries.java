package umontreal.ssj.mcqmctools;

import java.util.ArrayList;

import umontreal.ssj.functionfit.LeastSquares;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.stat.PgfDataTable;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.Chrono;
import umontreal.ssj.util.PrintfFormat;

/**
 * Basically, just a copy of #RQMCExperimentSeries but tailored for QMC. As a
 * matter of fact, it does not differ from RQMC with one repetition. But when we do actually know the mean, which
 * can be useful for toy models, this class offers tools to compute the integration error and estimate the convergence
 * rate of the error.
 * @author puchhamf
 *
 */
public class QMCExperimentSeries {

	int numSets = 0;   // Number of point sets in the series.
    PointSet[] theSets;   
    double base = 2.0;    // Base for the logs (in base 2 by default)
    double logOfBase;       // Math.log(base)
	double[] size = new double[numSets];    // values of n
	double[] mean = new double[numSets];    // average performance for each point set 
	double[] error = new double[numSets];
	double[] logn = new double[numSets];   // log_base n 
	double[] logError = new double[numSets]; // log_base (mean)
	String[] tableFields = {"n", "mean", "error", "log(n)", "log(error)"};
	                                       // Names of fields for table.
	boolean displayExec = false;   // When true, prints a display of execution in real time
	MonteCarloModelDouble model;
	// int numSkipRegression = 0; // Number of values of n that are skipped for the regression
	String cpuTime;       // time for last experiment\
    String title;
    
    double trueMean;
    
    
    /**
     * Constructor with a give series of QMC point sets.
     *  @param theSets      the QMC point sets.
     *  @param base 		the base used for all logarithms.
     *  @param mean 		the true mean for this experiment.
     */
    public QMCExperimentSeries (PointSet[] theSets, double base, double mean) {
 	   init(theSets, base);
 	   trueMean = mean;
    }
    
    /**
     * Constructor with a give series of RQMC point sets.
     *  @param theSets      the QMC point sets.
     *  @param base 		the base used for all logarithms.
     */
    public QMCExperimentSeries (PointSet[] theSets, double base) {
 	   init(theSets, base);
    }
    
    /**
     * Resets the array of QMC point sets for this object, and initializes 
     * (re-creates) the arrays that will contain the results.
     */
    public void init(PointSet[] theSets, double base) {
 	   this.base = base;
 	   this.logOfBase = Math.log(base);
 	     this.numSets = theSets.length;
 	     this.theSets = theSets;
 		 size = new double[numSets]; //  n for each point set
 	     mean = new double[numSets]; //  average for each point set
 	    error = new double[numSets]; //  error for each point set
 	     logn = new double[numSets];    // log n
 	     logError = new double[numSets];  // log (error)
    }
    
    /**
     * When set to true, a real-time display of execution results and CPU times 
     * will be printed on the default output.
     */
    public void setExecutionDisplay (boolean display) {
 	      this.displayExec = display;
    }
    
    /**
     * Sets the base for the logs to b.
     */
    public void setBase (double b) {
       base = b;
 	  logOfBase = Math.log(base);
    }

    /**
     * Returns the base used for the logs.
     */
    public double getBase() {
       return base;
    }

    /**
     * Returns the point set number i associated to this object (starts at 0).
     *  @return the ith point set associated to this object
     */
    public PointSet getSet(int i) {
       return theSets[i];
    }

    /**
     * Returns the vector of values of n, after an experiment.
     */
    public double[] getValuesn() {
       return size;
    }

    /**
     * Returns the vector of log_base(n), after an experiment.
     */
    public double[] getLogn() {
       return logn;
    }

   /**
     * Returns the vector of means from last experiment, after an experiment.
     */
    public double[] getMeans() {
       return mean;
    }
    
    public double[] getErrors() {
        return error;
     }

     /**
      * Returns the vector of log_base of error from the last experiment.
      */
     public double[] getLogErrors() {
        return logError;
     }

     /**
      * Performs an RQMC experiment with the given model, with this series of RQMC point sets.  
      * For each set in the series, computes m replicates of the RQMC estimator, 
      * the computes the average and the variance of these m replicates, 
      * and the logs of n and of the variance in the given base.
      */
     public void testErrorRate (MonteCarloModelDouble model) {
  		int n;
  		Tally statReps = new Tally();
  		Chrono timer = new Chrono();
  		this.model = model;
  	    if (displayExec) {
  			System.out.println("\n ============================================= ");
  	    	System.out.println("QMC simulation for mean estimation:  ");
  	    	System.out.println("Model: " + model.toString());
//  	    	System.out.println(" Number of indep copies m  = " + m);
//  	    	System.out.println(" Point sets: " + theSets[0].toString() + "\n");
  	    	System.out.println(" Point sets: " + theSets[0].toString() + "\n");
  			System.out.println(" n \tCPU time \tmean \t\tlog(var) ");	    	
  	    }
//  	    System.out.println("   log(n)\tmean\tlog(variance)\t\n");
  		
  		for (int s = 0; s < numSets; s++) { // For each cardinality n
  			n = theSets[s].getNumPoints();
  			size[s] = n;
  			logn[s] = Math.log(n) / logOfBase;
  			// System.out.println(" n = " + n + ", log n = " + logn[s] + "\n"); // ****
  			// System.out.println("  " + n + "     " + timer.format());
  			MonteCarloExperiment.simulateRuns (model,n ,theSets[s].iterator(), statReps);
  			mean[s] = statReps.average();
  			error[s] = Math.abs(statReps.average() - trueMean);
  			
  		    logError[s] = Math.log(error[s]) / logOfBase;
//  		    System.out.println(" " + logn[s] + "\t " + PrintfFormat.f(10, 5, mean[s]) + "\t "
//  			        + PrintfFormat.f(10, 5, logVar[s]) + "\n");
  		    if (displayExec) {
  			   System.out.println(" " + n + "\t" + timer.format() + 
  			              "\t" + PrintfFormat.f(10, 5, mean[s]) + 
  			              "\t" + PrintfFormat.f(7, 2, logError[s]));
  		    }
  		}	   
          cpuTime = timer.format();	 
     }
     
     /**
      * Performs a linear regression of log(variance) vs log(n), and returns the 
      * coefficients (constant and slope) in two-dimensional vector.
      * The first numSkip values in the array are skipped (not used) to make the regression.
      * This is useful if we want to focus the regression on larger values of n. 
      */
     public double[] regressionLogError (int numSkip) {
  		double[] x2 = new double[numSets-numSkip], y2 = new double[numSets-numSkip];
  		for (int i = 0; i < numSets-numSkip; ++i) {
  			x2[i] = logn[i+numSkip];
  			y2[i] = logError[i+numSkip];
  		}
  		return LeastSquares.calcCoefficients(x2, y2, 1);
  	}
     
     /**
      * Takes the regression coefficients of log(variance) in #regCoeff and returns a two-line string 
      * that reports on these coefficients.  
      * @param  regCoeff  the regression coefficients.
      * @return  Report as a string.
      */
      public String formatRegression (double[] regCoeff) {
  		StringBuffer sb = new StringBuffer("");
  		// double[] regCoeff = regressionLogVariance (numSkipRegression);
  		sb.append("  Slope of log(error) = " + PrintfFormat.f(8, 5, regCoeff[1]) + "\n");
  		sb.append("    constant term      = " + PrintfFormat.f(8, 5, regCoeff[0]) + "\n\n");
  		return sb.toString();
  	}
      
      /**
       * Produces and returns a report on the last experiment.
       * @param numSkip  The first numSkip values of n are skipped for the regression
       * @param details  If true, gives values (mean, log variance,...) for each n.
       * @return  Report as a string.
       */
   	public String reportErrorRate (int numSkip, boolean details) {
   		StringBuffer sb = new StringBuffer("");
   		sb.append("\n ============================================= \n");
   		sb.append("QMC simulation for mean estimation: \n ");
   		sb.append("Model: " + model.toString() + "\n");
   		;
   		sb.append("\tQMC point sets: " + theSets[0].toString() + "\n\n");
   		sb.append("QMC error \n");
   		if (details) sb.append(dataLogForPlot());
   		sb.append (formatRegression (regressionLogError (numSkip)));
   		// sb.append("  Slope of log(var) = " + PrintfFormat.f(8, 5, regCoeff[1]) + "\n");
   		// sb.append("    constant term      = " + PrintfFormat.f(8, 5, regCoeff[0]) + "\n\n");
   		sb.append("  Total CPU Time = " + cpuTime + "\n");
   		sb.append("-----------------------------------------------------\n");		
   		return sb.toString();
   	}
   	
   	/**
	 * Takes the data from the most recent experiment and returns it in a @ref PgfDataTable.
	 * This will typically be used to plot the data.
	 * 
	 * @param tableName  Name (short identifier) of the table.
	 * @return Report as a string.
	 */
	public PgfDataTable toPgfDataTable(String tableName, String tableLabel) {
        double[][] data = new double[numSets][5];
		for (int s = 0; s < numSets; s++) { // For each cardinality n
			data[s][0] = size[s];
		    data[s][1] = mean[s];
		    data[s][2] = error[s];
		    data[s][3] = logn[s];
		    data[s][4] = logError[s];
		}
		return new PgfDataTable (tableName, tableLabel, tableFields, data);
	}

	public PgfDataTable toPgfDataTable(String tableLabel) {
		return toPgfDataTable (title, tableLabel);
	}
	
	/**
	 * Returns the data on the mean and variance for each n, in an appropriate format to produce a
	 * plot with the pgfplot package.
	 * 
	 * @return Report as a string.
	 */
	public String dataForPlot() {
		StringBuffer sb = new StringBuffer("");
		sb.append("    n      mean       error \n");
		for (int s = 0; s < numSets; s++)  // For each cardinality n
			sb.append(" " + size[s] + " " + PrintfFormat.f(10, 5, mean[s]) + " "
			        + PrintfFormat.f(10, 5, error[s]) + "\n");
		return sb.toString();
	}

	/**
	 * Similar to dataForPlot, but for the log(variance) in terms of log n.
	 * 
	 * @return Report as a string.
	 */
	public String dataLogForPlot() {
		StringBuffer sb = new StringBuffer("");
		sb.append("   log(n)\tmean\tlog(error)\t\n");
		for (int s = 0; s < numSets; s++)  // For each cardinality n
			sb.append(" " + logn[s] + "\t " + PrintfFormat.f(10, 5, mean[s]) + "\t "
			        + PrintfFormat.f(10, 5, logError[s]) + "\n");
		return sb.toString();
	}
	
	/**
	 * Performs an experiment (testVarianceRate) for each point set series in the given list,
	 * and returns a report as a string. 
	 * 
	 * @param model
	 * @param list
	 * @param m
	 * @return  a report on the experiment.
	 */
	public String testErrorRateManyPointTypes (MonteCarloModelDouble model, 
			ArrayList<PointSet[]> list,
			int numSkip, 
			boolean makePgfTable, boolean printReport, boolean details,
			ArrayList<PgfDataTable> listCurves) {
		StringBuffer sb = new StringBuffer("");
	    // if (makePgfTable)  
	    //	listCurves = new ArrayList<PgfDataTable>();
		for(PointSet[] ptSeries : list) {
			init (ptSeries, base);
			testErrorRate (model);
			if (printReport)  System.out.println(reportErrorRate (numSkip, details));			
			if (printReport) sb.append (reportErrorRate (numSkip, details));			
            if (makePgfTable == true)  listCurves.add (toPgfDataTable 
            		(ptSeries[0].toString()));
		}
		return sb.toString();
	}
}
