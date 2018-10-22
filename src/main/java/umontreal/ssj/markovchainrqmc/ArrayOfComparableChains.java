package umontreal.ssj.markovchainrqmc;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import umontreal.ssj.functionfit.LeastSquares;
import umontreal.ssj.hups.CachedPointSet;
import umontreal.ssj.hups.PointSet;
import umontreal.ssj.hups.PointSetIterator;
import umontreal.ssj.hups.PointSetRandomization;
import umontreal.ssj.hups.RQMCPointSet;
import umontreal.ssj.stat.PgfDataTable;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.Chrono;
import umontreal.ssj.util.Num;
import umontreal.ssj.util.PrintfFormat;
import umontreal.ssj.util.sort.MultiDim;
import umontreal.ssj.util.sort.MultiDim01;
import umontreal.ssj.util.sort.MultiDimSort;
import umontreal.ssj.util.sort.MultiDimSort01;
import umontreal.ssj.util.sort.NeuralNetworkSort;

/**
 * This class provides tools to simulate an array of
 * 
 * @ref MarkovChainComparable objects with the array-RQMC method of
 * @cite vLEC08a, @cite vLEC09d&thinsp;. It offers tools to construct the array
 *       of chains, simulate with array-RQMC for one step or multiple steps,
 *       perform experiments, and report some results.
 *
 *       At each step, the @f$n@f$ states (which correspond to the @f$n@f$
 *       copies of the chain) are sorted in @f$\ell@f$ dimensions (usually the
 *       dimension of the state). If @f$\ell=1@f$, this is an ordinary sort.
 *       If @f$\ell> 1@f$, they are sorted using a @ref
 *       umontreal.ssj.util.MultiDimSort, which in general can sort @ref
 *       umontreal.ssj.util.MultiDimComparable objects. As a special case, they
 *       can also be sorted using a
 * @ref umontreal.ssj.util.MultiDimSort01 type of sort. Some of these sorts
 *      effectively map the chain states to the one-dimensional interval
 * @f$(0,1)@f$. For example, the sorts based on a Hilbert curve, such as
 * @ref umontreal.ssj.util.HilbertCurveSort, do that. In that case, we put
 * @f$\ell’=1@f$, otherwise @f$\ell’=\ell@f$.
 *
 *                To move the chains ahead, we use
 *                an @f$(\ell’+d)@f$-dimensional RQMC
 * @ref umontreal.ssj.hups.PointSet. The first @f$\ell’@f$ coordinates are used
 *      to sort the points to map them to the (sorted) chains. In some cases,
 *      when @f$\ell’=1@f$, the first coordinate of the points is not stored
 *      explicitly; the points are already sorted by a first coordinate which is
 *      only implicit (it serves to enumerate the points). This is what happens
 *      for a Sobol’ sequence, or a @ref umontreal.ssj.hups.KorobovLattice for
 *      which the first coordinate is dropped, for example. Only @f$d@f$
 *      coordinates are produced with those point sets and have to be
 *      randomized. In the other cases, the points must be sorted by their
 *      first @f$\ell’@f$ coordinates.
 *
 *      At each step the @ref umontreal.ssj.hups.PointSet is randomized by using
 *      a @ref umontreal.ssj.hups.PointSetRandomization. There are types of
 *      point sets for which only the last @f$d@f$ coordinates have to be
 *      randomized (e.g., digital nets and lattice rules) and others for which
 *      all @f$\ell’ + d@f$ coordinates must be randomized (e.g., a stratified
 *      sample).
 *
 *      When applying the array-RQMC method, the number of point coordinates
 *      used explicitly to sort the points and match the chains (which is either
 * @f$\ell’@f$ or 0) is passed in a variable named `sortCoord`, and the number
 *             of these coordinates that must be randomized at each step is
 *             passed in a variable named `sortCoordRand`. In the case
 *             where @f$\ell’=1@f$ and the first coordinate is only implicit,
 *             both should be 0. In the case where @f$\ell’ > 1@f$ but only the
 *             last @f$d@f$ coordinates have to be randomized at each step, one
 *             would put `sortCoord` to @f$\ell’@f$ and `sortCoordRand` to 0.
 *
 *             <div class="SSJ-bigskip"></div><div class="SSJ-bigskip"></div>
 */
public class ArrayOfComparableChains<T extends MarkovChainComparable> {
	protected T baseChain; // The base chain.
	protected int n; // Current number of chains.
	// protected int stateDim; // Dimension of the chain state.
	protected T[] chains; // Array of n comparable chains.
	protected double[] performances; // Performances for the n chains.
	protected double[][] states;
	protected PointSetRandomization randomization;
	protected MultiDimSort<T> savedSort;
	protected int sortCoordPts = 0; // Point coordinates used to sort points.
	double[][] object;
	static String PerformanceFile;
	static String filename;
	static String[] ArrayOfFilePerformance;
	public static final int seed = 12345;
	// Number of epochs (full passes of the data)
	public static final int nEpochs = 30;
	public static final Random rng = new Random(seed);
	// Batch size: i.e., each epoch has nSamples/batchSize parameter updates
	// public static final int batchSize = 1000;
	// Network learning rate
	public static final double learningRate = 0.06;
	static int numHidden = 10;
	public static String label;

	/**
	 * Creates an array of the comparable chain `baseChain`. The method
	 * #makeCopies(int) must be called afterward to make the actual copies of the
	 * chain.
	 */
	public ArrayOfComparableChains(T baseChain) {
		this.baseChain = baseChain;
		// stateDim = baseChain.stateDim;
	}

	/**
	 * Creates an array of the comparable chain `baseChain`. The method
	 * #makeCopies(int) must be called to make the copies. `rand` will be used to
	 * randomize the point sets in the simulations. `sort` will be used to sort the
	 * chains.
	 */
	public ArrayOfComparableChains(T baseChain, PointSetRandomization rand, MultiDimSort sort) {
		this.baseChain = baseChain;
		// stateDim = baseChain.stateDim;
		randomization = rand;
		savedSort = sort;
	}

	/**
	 * Creates <tt>n</tt> copies (clones) of the chain <tt>baseChain</tt> and puts
	 * them in an array, ready for the array RQMC simulation.
	 */
	public void makeCopies(int n) {

		@SuppressWarnings("unchecked")
		final T[] c = (T[]) Array.newInstance(baseChain.getClass(), n);
		chains = c; // Array of Markov chains.

		this.n = n;
		performances = new double[n]; // Array to store the performances.
		for (int i = 0; i < n; i++) {
			try {
				chains[i] = (T) baseChain.clone();

			} catch (CloneNotSupportedException e) {
				System.err.println("ArrayOfComparableChains:");
				e.printStackTrace();
			}
		}
	}

	/**
	 * Initializes the <tt>n</tt> copies (clones) of the chain <tt>baseChain</tt> to
	 * their initial state by calling
	 * {@link umontreal.ssj.markovchain.MarkovChain.initialState() initialState()}
	 * on each chain.
	 */
	public void initialStates() {
		int i = 0;
		for (T mc : chains) {
			mc.initialState();
			performances[i] = mc.getPerformance(); // Needed? Why do this?
			++i;
		}
	}

	/**
	 * Returns the number `n` of chains.
	 */
	public int getN() {
		return n;
	}

	/**
	 * Returns the underlying array of `n` @ref MarkovChainComparable.
	 */
	public T[] getChains() {
		return chains;
	}

	/**
	 * Sets the internal @ref umontreal.ssj.hups.PointSetRandomization to `rand`.
	 */
	public void setRandomization(PointSetRandomization rand) {
		randomization = rand;
	}

	/**
	 * Returns the internal @ref umontreal.ssj.hups.PointSetRandomization.
	 */
	public PointSetRandomization getRandomization() {
		return randomization;
	}

	/**
	 * Sets the internal @ref umontreal.ssj.util.MultiDimSort to `sort`.
	 */
	public void setSort(MultiDimSort sort) {
		savedSort = sort;
	}

	/**
	 * Returns the saved @ref umontreal.ssj.util.MultiDimSort.
	 */
	public MultiDimSort getSort() {
		return savedSort;
	}

	/**
	 * Randomized the point set `p` and Simulates the @f$n@f$ copies of the chain,
	 * one step for each copy, using
	 * 
	 * @ref umontreal.ssj.hups.PointSet `p`, where @f$n@f$ is the current number of
	 *      copies (clones) of the chain and is *assumed* to equal the number of
	 *      points in `p`. The points are randomized before the simulation using the
	 *      stored
	 * @ref umontreal.ssj.hups.PointSetRandomization. If `sortCoordPts`
	 * @f$>0@f$, the points are also sorted explicitly based on their first
	 *           `sortCoordPts` coordinates, at each step, after they are
	 *           randomized. In that case, `p` must implement the
	 * @ref umontreal.ssj.util.MultiDim01 interface. The dimension of `p` must be at
	 *      least as large as `sortCoordPts` @f$+@f$ the number of uniforms required
	 *      to simulate one step of the chain. Returns the number of chains that
	 *      have not stopped yet.
	 */
	public int simulOneStepArrayRQMC(PointSet p, PointSetRandomization rand, MultiDimSort sort, int sortCoordPts) {
		int nStopped = 0;
		p.randomize(rand); // Randomize point set.
		if (sortCoordPts > 0) {
			if (!(p instanceof CachedPointSet))
				throw new IllegalArgumentException("p is not a CachedPointSet.");
			if (sortCoordPts > 1)
				((CachedPointSet) p).sort(sort); // Sort points using first sortCoordPts coordinates.
			else
				((CachedPointSet) p).sortByCoordinate(0); // Sort by first coordinate.
		}
		PointSetIterator stream = p.iterator();
		stream.resetCurPointIndex(); // Go to first point.
		int i = 0;
		for (T mc : chains) { // Assume the chains are sorted
			if (mc.hasStopped()) {
				++nStopped;
			} else {
				stream.setCurCoordIndex(sortCoordPts); // Skip first sortCoordPts coord.
				mc.nextStep(stream); // simulate next step of the chain.
				stream.resetNextSubstream(); // Go to next point.
				if (mc.hasStopped())
					++nStopped;
			}
			performances[i] = mc.getPerformance();
			++i;
		}
		return n - nStopped;
	}

	/**
	 * This version uses the preselected randomization and sort, with `sortCoordPts
	 * = 0`.
	 */
	public int simulOneStepArrayRQMC(PointSet p) {
		return simulOneStepArrayRQMC(p, randomization, savedSort, 0);
	}
	/*
	 * public void WriteExcelFile (Object[][] object, int step) { HSSFWorkbook
	 * workbook = new HSSFWorkbook();
	 * 
	 * // Create a blank sheet HSSFSheet sheet =
	 * workbook.createSheet("Performance Markov Chain"+step);
	 * 
	 * // This data needs to be written (Object[]) Map<Integer, Object[]> data = new
	 * TreeMap<Integer, Object[] >();
	 * 
	 * data.put(0, new Object[]{ "Xi", "Performance" }); for ( int i=0; i<
	 * object.length ; i++) data.put(i+1, object[i]);
	 * 
	 * 
	 * Set<Integer> keyset = data.keySet(); int rownum = 0; for (Integer key :
	 * keyset) { // this creates a new row in the sheet Row row =
	 * sheet.createRow(rownum++); Object[] objArr = data.get(key); int cellnum = 0;
	 * for (Object obj : objArr) { // this line creates a cell in the next column of
	 * that row Cell cell = row.createCell(cellnum++); if (obj instanceof String)
	 * cell.setCellValue((String)obj); else if (obj instanceof Double)
	 * cell.setCellValue((Double)obj); else if (obj instanceof Integer)
	 * cell.setCellValue((Integer)obj); } } String PerformanceFile =
	 * "Excelperformance"+step+".xls"; try { FileOutputStream out = new
	 * FileOutputStream(new File(PerformanceFile)); workbook.write(out);
	 * out.close(); } catch (Exception e) { e.printStackTrace(); } }
	 */
	/*
	 * public void ReadExcelFile (String filename) { try { FileInputStream file =
	 * new FileInputStream(new File(filename));
	 * 
	 * //Create Workbook instance holding reference to .xlsx file HSSFWorkbook
	 * workbook = new HSSFWorkbook(file);
	 * 
	 * //Get first/desired sheet from the workbook HSSFSheet sheet =
	 * workbook.getSheetAt(0);
	 * 
	 * //Iterate through each rows one by one Iterator<Row> rowIterator =
	 * sheet.iterator(); while (rowIterator.hasNext()) { Row row =
	 * rowIterator.next(); //For each row, iterate through all the columns
	 * Iterator<Cell> cellIterator = row.cellIterator();
	 * 
	 * while (cellIterator.hasNext()) { Cell cell = cellIterator.next(); //Check the
	 * cell type and format accordingly switch (cell.getCellType()) { case
	 * Cell.CELL_TYPE_NUMERIC: System.out.print(cell.getNumericCellValue() + "\t");
	 * break; case Cell.CELL_TYPE_STRING: System.out.print(cell.getStringCellValue()
	 * + "\t"); break; } } System.out.println(""); } file.close(); } catch
	 * (Exception e) { e.printStackTrace(); } }
	 */
	/*
	 * public static void addSheet(HSSFSheet mergedSheet, HSSFSheet sheet) { // map
	 * for cell styles Map<Integer, HSSFCellStyle> styleMap = new HashMap<Integer,
	 * HSSFCellStyle>();
	 * 
	 * // This parameter is for appending sheet rows to mergedSheet in the end int
	 * len = mergedSheet.getLastRowNum(); for (int j = sheet.getFirstRowNum(); j <=
	 * sheet.getLastRowNum(); j++) {
	 * 
	 * HSSFRow row = sheet.getRow(j); HSSFRow mrow = mergedSheet.createRow(len + j +
	 * 1);
	 * 
	 * for (int k = row.getFirstCellNum(); k < row.getLastCellNum(); k++) { HSSFCell
	 * cell = row.getCell(k); HSSFCell mcell = mrow.createCell(k);
	 * 
	 * if (cell.getSheet().getWorkbook() == mcell.getSheet() .getWorkbook()) {
	 * mcell.setCellStyle(cell.getCellStyle()); } else { int stHashCode =
	 * cell.getCellStyle().hashCode(); HSSFCellStyle newCellStyle =
	 * styleMap.get(stHashCode); if (newCellStyle == null) { newCellStyle =
	 * mcell.getSheet().getWorkbook() .createCellStyle();
	 * newCellStyle.cloneStyleFrom(cell.getCellStyle()); styleMap.put(stHashCode,
	 * newCellStyle); } mcell.setCellStyle(newCellStyle); }
	 * 
	 * switch (cell.getCellType()) { case HSSFCell.CELL_TYPE_FORMULA:
	 * mcell.setCellFormula(cell.getCellFormula()); break; case
	 * HSSFCell.CELL_TYPE_NUMERIC: mcell.setCellValue(cell.getNumericCellValue());
	 * break; case HSSFCell.CELL_TYPE_STRING:
	 * mcell.setCellValue(cell.getStringCellValue()); break; case
	 * HSSFCell.CELL_TYPE_BLANK: mcell.setCellType(HSSFCell.CELL_TYPE_BLANK); break;
	 * case HSSFCell.CELL_TYPE_BOOLEAN:
	 * mcell.setCellValue(cell.getBooleanCellValue()); break; case
	 * HSSFCell.CELL_TYPE_ERROR: mcell.setCellErrorValue(cell.getErrorCellValue());
	 * break; default: mcell.setCellValue(cell.getStringCellValue()); break; } } } }
	 */
	/*
	 * public static void MergeFile(String Excelperformance1,String
	 * Excelperformance2 ) { try { // excel files FileInputStream excellFile1 = new
	 * FileInputStream(new File( Excelperformance1)); FileInputStream excellFile2 =
	 * new FileInputStream(new File( Excelperformance2));
	 * 
	 * 
	 * // Create Workbook instance holding reference to .xlsx file HSSFWorkbook
	 * workbook1 = new HSSFWorkbook(excellFile1); HSSFWorkbook workbook2 = new
	 * HSSFWorkbook(excellFile2);
	 * 
	 * // Get first/desired sheet from the workbook HSSFSheet sheet1 =
	 * workbook1.getSheetAt(0); HSSFSheet sheet2 = workbook2.getSheetAt(0);
	 * 
	 * // add sheet2 to sheet1 addSheet(sheet1, sheet2); excellFile1.close();
	 * 
	 * // save merged file File mergedFile = new File( "MergedPerformance.xls"); if
	 * (!mergedFile.exists()) { mergedFile.createNewFile(); } FileOutputStream out =
	 * new FileOutputStream(mergedFile); workbook1.write(out);
	 * 
	 * out.close(); System.out .println("Files were merged succussfully"); } catch
	 * (Exception e) { e.printStackTrace(); }
	 * 
	 * }
	 */

	/**
	 * Simulates the @f$n@f$ copies of the chain, `numSteps` steps for each copy,
	 * using @ref umontreal.ssj.hups.PointSet `p`, where @f$n@f$ is the current
	 * number of copies (clones) of the chain and is *assumed* to equal the number
	 * of points in `p`. At each step, the points are randomized using `rand`. All
	 * coordinates are randomized. If `sortCoordPts` @f$>0@f$, the points are also
	 * sorted explicitly based on their first `sortCoordPts` coordinates, at each
	 * step, after they are randomized. In that case, `p` must implement the
	 * 
	 * @ref umontreal.ssj.util.MultiDim01 interface. If the coordinates used for the
	 *      sort do not have to be randomized at each step and the points do not
	 *      have to be sorted again, one should remove these coordinates before
	 *      invoking this method and use `sortCoordPts=0`. In this case, the points
	 *      must be sorted before invoking this method. The class @ref
	 *      umontreal.ssj.hups.SortedAndCutPointSet can be useful for this. The
	 *      dimension of `p` must be at least as large as `sortCoordPts` @f$+@f$ the
	 *      number of uniforms required to simulate one step of the chain. The
	 *      method returns the average performance per run. An array that contains
	 *      the performance for each run can also be obtained via
	 *      #getPerformances()(.)
	 */
	/*
	 * public double simulArrayRQMC (PointSet p, PointSetRandomization rand,
	 * MultiDimSort sort, int sortCoordPts, double numSteps) { object = new
	 * Object[n][1+p.getDimension()]; states = new double[n][]; int numNotStopped =
	 * n; initialStates(); int step = 0; while (step < numSteps && numNotStopped >
	 * 0) { if (numNotStopped == n) sort.sort(chains, 0, n); else
	 * sortNotStoppedChains (sort); // Sort the numNotStopped first chains.
	 * p.randomize(rand); // Randomize the point set. if (sortCoordPts > 0) { if
	 * (!(p instanceof CachedPointSet)) throw new
	 * IllegalArgumentException("p is not a CachedPointSet."); if (sortCoordPts > 1)
	 * ((CachedPointSet) p).sort(sort); // Sort points using first sortCoordPts
	 * coordinates. else ((CachedPointSet) p).sortByCoordinate (0); // Sort by first
	 * coordinate. } PointSetIterator stream = p.iterator ();
	 * stream.resetCurPointIndex (); // Go to first point. int i = 0; for (T mc :
	 * chains) { // Assume the chains are sorted if (mc.hasStopped()) {
	 * numNotStopped--; } else { stream.setCurCoordIndex (sortCoordPts); // Skip
	 * first sortCoordPts coord. mc.nextStep (stream); // simulate next step of the
	 * chain. stream.resetNextSubstream (); // Go to next point. if
	 * (mc.hasStopped()) numNotStopped--; } performances[i] = mc.getPerformance();
	 * states[i] = mc.getState(); object [i][0] = performances[i]; int t=1; for (
	 * int j=0;j<states[i].length;j++){ object[i][t] = states[i][j]; t++; } // if
	 * (step== numSteps) ExcelFile (object, step) ; ++i; } ++step; } return
	 * calcMeanPerf(); }
	 */

	/*
	 * public static double[][] transposeMatrix(double [][] m){ double [][] tmp=new
	 * double[m[0].length][m.length]; for (int i=0; i< m[0].length; i++) for (int
	 * j=0; j< m.length; j++) tmp[i][j] = m[j][i]; return tmp; }
	 */
	/*
	 * public static DataSetIterator generateDataSet( String filename, int
	 * batchSize) throws FileNotFoundException {
	 * 
	 * FileReader file = new FileReader(filename); Scanner scanner = new
	 * Scanner(file); int count = 0; while (scanner.hasNextLine()) {
	 * scanner.nextLine(); count++; } String line = ""; double [] res; double [][]
	 * inputs = new double[count][]; double [] output = new double[count]; int l=0;
	 * try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
	 * 
	 * while ((line = br.readLine()) != null) { String[] country = line.split(",");
	 * 
	 * res= new double[country.length-1]; for(int i=0; i<country.length-1;i++){
	 * res[i] = Double.parseDouble(country[i]); inputs [l] = res; } output[l]=
	 * Double.parseDouble(country[country.length-1]);
	 * 
	 * l++; } } catch (IOException e) { e.printStackTrace(); } double [] [] inputt =
	 * new double[inputs[0].length][inputs.length]; inputt =
	 * transposeMatrix(inputs); System.out.println(" lines"+inputt.length);
	 * INDArray[] inputArray= new INDArray[inputt.length]; for (int i=0;i<
	 * inputArray.length; i++) inputArray[i] = Nd4j.create(inputt[i], new
	 * int[]{count,1});
	 * 
	 * 
	 * //INDArray inputNDArray = Nd4j.hstack(inputNDArray1,inputNDArray2); INDArray
	 * inputNDArray = Nd4j.hstack(inputArray);
	 * 
	 * System.out.println("NbrInput"+ inputNDArray .length());
	 * 
	 * INDArray outPut = Nd4j.create(output, new int[]{count, 1}); DataSet dataSet =
	 * new DataSet(inputNDArray, outPut); // List<DataSet> listDs =
	 * dataSet.asList(); java.util.List<DataSet> listDs = dataSet.asList();
	 * Collections.shuffle(listDs, rng); return new ListDataSetIterator(listDs,
	 * batchSize); } public MultiLayerNetwork getDeepDenseLayerNetworkConfiguration(
	 * int numInputs, int numOutputs, int numHiddenNodes , int seed, double
	 * learningRate) {
	 * 
	 * MultiLayerNetwork net = new MultiLayerNetwork(new
	 * NeuralNetConfiguration.Builder() .seed(seed) .weightInit(WeightInit.XAVIER)
	 * //.updater(new Nesterovs(learningRate, 0.9))
	 * .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	 * .updater(new org.nd4j.linalg.learning.config.Nesterovs(0.006,0.9))
	 * //.updater(new Nesterovs(learningRate,0.9)) // .updater(new
	 * org.nd4j.linalg.learning.config.Nesterovs(learningRate,0.9)) .list()
	 * .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
	 * .activation(Activation.TANH).build()) .layer(1, new
	 * DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
	 * .activation(Activation.TANH).build()) .layer(2, new
	 * OutputLayer.Builder(LossFunctions.LossFunction.MSE)
	 * .activation(Activation.IDENTITY)
	 * .nIn(numHiddenNodes).nOut(numOutputs).build())
	 * .pretrain(false).backprop(true).build());
	 * 
	 * net.init(); net.setListeners(new ScoreIterationListener(1));
	 * 
	 * return net; }
	 */

	// MultiLayerConfiguration conf = getDeepDenseLayerNetworkConfiguration();
	/*
	 * public double simulArrayRQMC (PointSet p, PointSetRandomization rand,
	 * MultiDimSort sort, int sortCoordPts, double numSteps) { int stateDim=
	 * chains[0].stateDim; object = new double[n][chains[0].stateDim+1]; states =
	 * new double[n][]; char SEPARATOR = ','; int numNotStopped = n;
	 * initialStates(); int step = 0; if (sort.toString().equals("Neural")){
	 * 
	 * while (step < numSteps && numNotStopped > 0) {
	 * 
	 * PointSetIterator stream = p.iterator (); stream.resetCurPointIndex (); // Go
	 * to first point. int i = 0; for (T mc : chains) { // Assume the chains are
	 * sorted //object = new double[n][mc.getState().length+1]; if (mc.hasStopped())
	 * { numNotStopped--; } else { stream.setCurCoordIndex (sortCoordPts); // Skip
	 * first sortCoordPts coord. mc.nextStep (stream); // simulate next step of the
	 * chain. stream.resetNextSubstream (); // Go to next point. if
	 * (mc.hasStopped()) numNotStopped--; } performances[i] = mc.getPerformance();
	 * states[i] = mc.getState(); // object [i][0] = performances[i];
	 * 
	 * for ( int j=0;j<states[i].length;j++) object[i][j] = states[i][j];
	 * 
	 * // System.out.println("lenght"+ object[0].length);
	 * 
	 * object [i][states[i].length] = performances[i]; // if (step== numSteps)
	 * 
	 * try { PerformanceFile = "Excelperformance"+step+".csv"; FileWriter writer =
	 * new FileWriter(PerformanceFile);
	 * 
	 * writeLine(writer, object[i], SEPARATOR); writer.flush(); writer.close();
	 * ArrayOfFilePerformance[step] = PerformanceFile; } catch (Exception ignored)
	 * {}
	 * 
	 * 
	 * ++i; }
	 * 
	 * ++step; } //String filename ; int[] idx = new int[stateDim+1]; int
	 * id=stateDim; for (int i=0;i<idx.length; i++){ idx [i] = id; id =id+1; }
	 * 
	 * try { filename =combinestep(3, (int)numSteps-1); removecolumns(
	 * filename,idx); } catch (Exception ignored) {} String [] nameTrainTest =
	 * TrainTestFile(filename); String filenameTrain = nameTrainTest[0]; String
	 * filenameTest = nameTrainTest[1];
	 * 
	 * 
	 * File inputfile1= new File(ArrayOfFilePerformance[step]); File inputfile2= new
	 * File(ArrayOfFilePerformance[(int) (numSteps-1)]); File output= new
	 * File("FileF.csv"); try{ mergeFiles( output, inputfile1, inputfile2, ',') ; }
	 * catch (Exception ignored) {}
	 * 
	 * 
	 * 
	 * 
	 * } initialStates(); step = 0; while (step < numSteps && numNotStopped > 0) {
	 * 
	 * 
	 * if (numNotStopped == n) sort.sort(chains, 0, n); else sortNotStoppedChains
	 * (sort); // Sort the numNotStopped first chains. p.randomize(rand); //
	 * Randomize the point set. if (sortCoordPts > 0) { if (!(p instanceof
	 * CachedPointSet)) throw new
	 * IllegalArgumentException("p is not a CachedPointSet."); if (sortCoordPts > 1)
	 * ((CachedPointSet) p).sort(sort); // Sort points using first sortCoordPts
	 * coordinates. else ((CachedPointSet) p).sortByCoordinate (0); // Sort by first
	 * coordinate. } PointSetIterator stream = p.iterator ();
	 * stream.resetCurPointIndex (); // Go to first point. int i = 0; for (T mc :
	 * chains) { // Assume the chains are sorted if (mc.hasStopped()) {
	 * numNotStopped--; } else { stream.setCurCoordIndex (sortCoordPts); // Skip
	 * first sortCoordPts coord. mc.nextStep (stream); // simulate next step of the
	 * chain. stream.resetNextSubstream (); // Go to next point. if
	 * (mc.hasStopped()) numNotStopped--; } ++i; } ++step; } return calcMeanPerf();
	 * }
	 */

	private static final char DEFAULT_SEPARATOR = ',';

	public static void writeLine(Writer w, double[] values) throws IOException {
		writeLine(w, values, DEFAULT_SEPARATOR, ' ');
	}

	public static void writeLine(Writer w, double[] values, char separators) throws IOException {
		writeLine(w, values, separators, ' ');
	}

	public static void writeLine(Writer w, double[] values, char separators, char customQuote) throws IOException {

		boolean first = true;

		// default customQuote is empty

		if (separators == ' ') {
			separators = DEFAULT_SEPARATOR;
		}

		StringBuilder sb = new StringBuilder();
		/*
		 * int length=values.length; String[] header = new String[length]; for( int i=0;
		 * i< header.length-1;i++) header [i] ="X_["+i+"]";
		 * header[header.length-1]="Performance"; // sb.append(header); for (Object
		 * value : header) { sb.append(header); }
		 */
		for (double value : values) {
			if (!first) {
				sb.append(separators);
			}
			if (customQuote == ' ') {
				sb.append(value);
			} else {
				sb.append(customQuote).append(value).append(customQuote);
			}

			first = false;
		}
		sb.append("\n");
		w.append(sb.toString());

	}

	private static final void transfer(final Reader source, final Writer destination, char separator)
			throws IOException {
		char[] buffer = new char[1024 * 16];
		int len = 0;
		while ((len = source.read(buffer)) >= 0) {
			destination.write(buffer, 0, len);
		}
		destination.write(separator);
	}

	private static final void transfer(final Reader source, final Writer destination) throws IOException {
		char[] buffer = new char[1024 * 16];
		int len = 0;
		while ((len = source.read(buffer)) >= 0) {
			destination.write(buffer, 0, len);
		}

	}

	public static void mergeFiles(final File output, final File inputfile1, final File inputfile2, char separator)
			throws IOException {

		try (Reader sourceA = Files.newBufferedReader(inputfile1.toPath());
				Reader sourceB = Files.newBufferedReader(inputfile2.toPath());
				Writer destination = Files.newBufferedWriter(output.toPath(), StandardCharsets.UTF_8);) {

			transfer(sourceA, destination, separator);
			transfer(sourceB, destination);

		}
	}

	public void mergestep0(int step1, int step2) {
		File inputfile1 = new File(ArrayOfFilePerformance[(int) step1]);
		File inputfile2 = new File(ArrayOfFilePerformance[(int) step2]);

		File output = new File("FileF.csv");
		try {
			mergeFiles(output, inputfile1, inputfile2, ',');
		} catch (Exception ignored) {
		}
	}

	public String combinestep(int step1, int step2) throws IOException {

		BufferedReader br1 = new BufferedReader(
				new InputStreamReader(new FileInputStream(ArrayOfFilePerformance[(int) step1])));
		BufferedReader br2 = new BufferedReader(
				new InputStreamReader(new FileInputStream(ArrayOfFilePerformance[(int) step2])));
		BufferedWriter bw = new BufferedWriter(new FileWriter("Filecombination" + label + ".csv"));
		String line1, line2;
		while ((line1 = br1.readLine()) != null && (line2 = br2.readLine()) != null) {
			String line = line1 + "," + line2;
			String[] linesplit = line.split(",");

			double[] res = new double[linesplit.length];
			for (int i = 0; i < linesplit.length; i++) {
				res[i] = Double.parseDouble(linesplit[i]);
			}
			// bw.write(line);
			writeLine(bw, res, ',');
		}

		System.out.println("combination");
		bw.close();
		br1.close();
		br2.close();
		// filename= "Filecombination.csv";
		// return filename;
		return "Filecombination" + label + ".csv";
	}

	private static double[] remove(double[] a, int index) {
		if (a == null || index < 0 || index >= a.length) {
			return a;
		}

		double[] result = new double[a.length - 1];
		for (int i = 0; i < index; i++) {
			result[i] = a[i];
		}

		for (int i = index; i < a.length - 1; i++) {
			result[i] = a[i + 1];
		}

		return result;
	}

	public static double[] remove(double[] original, int removeStart, int removeEnd) {
		int originalLen = original.length;
		int length = originalLen - (removeEnd - removeStart + 1);
		double[] a = new double[length];
		for (int i = 0; i < removeStart; i++) {
			a[i] = original[i];
		}
		for (int i = removeStart; i < length; i++) {
			a[i] = original[i + removeEnd - removeStart + 1];
		}
		return a;
	}

	String removecolumns(String csvFile, int[] index, int step) throws IOException {

		// String csvFile = "File1.csv";
		String line = "";
		String cvsSplitBy = ",";
		for (int i = 0; i < index.length; i++)
			System.out.println("index" + index[i]);

		double[] resultt = null;
		String result = "result" + label + step + ".csv";
		// filename = result;
		FileWriter bw = new FileWriter(result);
		System.out.println("remove");
		try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
			int t = 0;
			while ((line = br.readLine()) != null) {
				String[] country = line.split(cvsSplitBy);
				Double[] ress = new Double[country.length];
				double[] res = new double[country.length];
				for (int i = 0; i < country.length; i++) {
					res[i] = Double.parseDouble(country[i]);
				}
				// System.out.println("length"+country.length);
				/*
				 * for (int i=0;i<index.length;i++) resultt = remove(res, index[i]);
				 */
				resultt = remove(res, index[0], index[index.length - 1]);
				// System.out.println("length"+resultt.length);
				writeLine(bw, resultt);
				t++;

			}
			System.out.println("nbr column" + resultt.length);
			System.out.println("nbr ligne" + t);
			bw.flush();
			bw.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		return result;
	}

	public String[] TrainTestFile(String inputfile) {

		FileWriter fstream1 = null;
		FileWriter fstream2 = null;
		try {
			// Reading file and getting no. of files to be generated
			// String inputfile = "File1.csv"; // Source File Name.
			// double nol = 11; // No. of lines to be split and saved in each output file.
			File file = new File(inputfile);
			Scanner scanner = new Scanner(file);
			int count = 0;
			while (scanner.hasNextLine()) {
				scanner.nextLine();
				count++;
			}
			System.out.println("Lines in the file: " + count); // Displays no. of lines in the input file.
			double nol1, nol2;
			int a = count % (10 / 8);
			System.out.println("a: " + a);
			if (a == 0) {
				nol1 = count * 0.8;
			} else
				nol1 = count * 0.8 + 1;

			int b = count % (10 / 2);
			System.out.println("b: " + b);
			if (b == 0) {
				nol2 = count * 0.2;
			} else
				nol2 = count * 0.2 + 1;

			System.out.println("Lines in the first file: " + nol1);
			System.out.println("Lines in the second file: " + nol2);
			// ---------------------------------------------------------------------------------------------------------
			// Actual splitting of file into smaller files

			FileInputStream fstream = new FileInputStream(inputfile);
			DataInputStream in = new DataInputStream(fstream);

			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			/*
			 * File f = new File("/u/benabama/TrainDataF.csv"); File f2 = new
			 * File("/u/benabama/TestDataF.csv"); fstream1 = new FileWriter(f); //
			 * Destination File Location BufferedWriter out1 = new BufferedWriter(fstream1);
			 * fstream2 = new FileWriter(f2); // Destination File Location
			 * System.out.println("creation"); BufferedWriter out2 = new
			 * BufferedWriter(fstream2);
			 */
			fstream1 = new FileWriter("TrainDataData" + label + ".csv"); // Destination File Location
			BufferedWriter out1 = new BufferedWriter(fstream1);
			fstream2 = new FileWriter("TestDataData" + label + ".csv"); // Destination File Location
			BufferedWriter out2 = new BufferedWriter(fstream2);
			for (int i = 1; i <= (int) nol1; i++) {
				strLine = br.readLine();
				if (strLine != null) {
					out1.write(strLine);
					if (i != nol1) {
						out1.newLine();
					}
				}
			}
			out1.close();

			// for (int i=(int) (nol1+1);i<=count-nol1;i++)
			for (int i = 1; i <= (int) count - (int) nol1; i++) {
				strLine = br.readLine();
				if (strLine != null) {
					out2.write(strLine);
					if (i != nol2) {
						out2.newLine();
					}
				}
			}
			out2.close();

			in.close();
		} catch (Exception e) {
			System.err.println("Error: " + e.getMessage());
		}

		String[] ret = { "TrainDataData" + label + ".csv", "TestDataData" + label + ".csv" };
		return ret;

	}

	public double simulArrayRQMC(PointSet p, PointSetRandomization rand, MultiDimSort sort, int sortCoordPts,
			double numSteps) {
		// String [] file =TrainTestFile(filename);

		// sort = new NeuralNetworkSort(chains[0].stateDim);

		// NeuralNetworkSort sort = new NeuralNetworkSort(chains[0].stateDim, file[0],
		// file[1],chains[0].stateDim, 1, 10, seed, learningRate, nEpochs );
		int stateDim = chains[0].stateDim;
		object = new double[n][chains[0].stateDim + 1];
		states = new double[n][];
		char SEPARATOR = ',';
		int step;
		int numNotStopped = n;

		if (sort.toString() == "NeuralNetwork") {
			initialStates();
			step = 0;

			ArrayOfFilePerformance = new String[(int) numSteps];
			while (step < numSteps && numNotStopped > 0) {
				PerformanceFile = "Excelperformance" + label + "_" + step + ".csv";
				FileWriter writer = null;
				try {
					writer = new FileWriter(PerformanceFile);
				} catch (IOException e) {

					e.printStackTrace();
				}
				PointSetIterator stream = p.iterator();
				stream.resetCurPointIndex(); // Go to first point.
				int i = 0;
				for (T mc : chains) { // Assume the chains are sorted
					// object = new double[n][mc.getState().length+1];
					if (mc.hasStopped()) {
						numNotStopped--;
					} else {
						stream.setCurCoordIndex(sortCoordPts); // Skip first sortCoordPts coord.
						mc.nextStep(stream); // simulate next step of the chain.
						stream.resetNextSubstream(); // Go to next point.
						if (mc.hasStopped())
							numNotStopped--;
					}
					performances[i] = mc.getPerformance();
					states[i] = mc.getState();
					// object [i][0] = performances[i];

					for (int j = 0; j < states[i].length; j++)
						object[i][j] = states[i][j];

					// System.out.println("lenght"+ object[0].length);

					object[i][states[i].length] = performances[i];
					// if (step== numSteps)

					try {

						writeLine(writer, object[i], SEPARATOR);

					} catch (Exception ignored) {
					}

					++i;

				}
				try {
					writer.flush();
					writer.close();
				} catch (IOException e) {

					e.printStackTrace();
				}
				ArrayOfFilePerformance[step] = PerformanceFile;

				++step;
			}

			int[] idx = new int[stateDim + 1];
			int id = stateDim;
			for (int i = 0; i < idx.length; i++) {
				idx[i] = id;
				id = id + 1;
			}
			/*
			 * for (int i=0; i<idx.length;i++) System.out.println("index"+idx[i]);
			 */
			String file = null;
			String filena = null;
			/*
			 * try { filena =combinestep(1, (int)numSteps-1); file = removecolumns(
			 * filena,idx); System.out.println("filename"+filena); } catch (Exception
			 * ignored) {} System.out.println("file"+file); String [] nameTrainTest =
			 * TrainTestFile(file); String filenameTrain = nameTrainTest[0]; String
			 * filenameTest = nameTrainTest[1];
			 * 
			 * sort = new NeuralNetworkSort2(chains[0].stateDim, filenameTrain,
			 * filenameTest,chains[0].stateDim, 1, numHidden, seed, learningRate, nEpochs );
			 */
			/*
			 * File inputfile1= new File(ArrayOfFilePerformance[step]); File inputfile2= new
			 * File(ArrayOfFilePerformance[(int) (numSteps-1)]); File output= new
			 * File("FileF.csv"); try{ mergeFiles( output, inputfile1, inputfile2, ',') ; }
			 * catch (Exception ignored) {}
			 */

			System.out.println("on est la");

			initialStates();
			step = 0;
			while (step < numSteps && numNotStopped > 0) {
				try {
					filena = combinestep(step, (int) numSteps - 1); 
					file = removecolumns(filena, idx, step);
					System.out.println("filename" + filena);
				} catch (Exception ignored) {
				}
				System.out.println("file" + file);
				String[] nameTrainTest = TrainTestFile(file);
				String filenameTrain = nameTrainTest[0];
				String filenameTest = nameTrainTest[1];

				sort = new NeuralNetworkSort(chains[0].stateDim, filenameTrain, filenameTest, chains[0].stateDim, 1,
						numHidden, seed, learningRate, nEpochs, 128);

				if (numNotStopped == n)
					sort.sort(chains, 0, n);
				else
					sortNotStoppedChains(sort); // Sort the numNotStopped first chains.
				p.randomize(rand); // Randomize the point set.
				if (sortCoordPts > 0) {
					if (!(p instanceof CachedPointSet))
						throw new IllegalArgumentException("p is not a CachedPointSet.");
					if (sortCoordPts > 1)
						((CachedPointSet) p).sort(sort); // Sort points using first sortCoordPts coordinates.
					else
						((CachedPointSet) p).sortByCoordinate(0); // Sort by first coordinate.
				}
				PointSetIterator stream = p.iterator();
				stream.resetCurPointIndex(); // Go to first point.
				int i = 0;
				for (T mc : chains) { // Assume the chains are sorted
					if (mc.hasStopped()) {
						numNotStopped--;
					} else {
						stream.setCurCoordIndex(sortCoordPts); // Skip first sortCoordPts coord.
						mc.nextStep(stream); // simulate next step of the chain.
						stream.resetNextSubstream(); // Go to next point.
						if (mc.hasStopped())
							numNotStopped--;
					}
					performances[i] = mc.getPerformance();
					++i;
				}
				++step;
			}
			return calcMeanPerf();
		} else {

			initialStates();
			step = 0;
			while (step < numSteps && numNotStopped > 0) {

				if (numNotStopped == n)
					sort.sort(chains, 0, n);
				else
					sortNotStoppedChains(sort); // Sort the numNotStopped first chains.
				p.randomize(rand); // Randomize the point set.
				if (sortCoordPts > 0) {
					if (!(p instanceof CachedPointSet))
						throw new IllegalArgumentException("p is not a CachedPointSet.");
					if (sortCoordPts > 1)
						((CachedPointSet) p).sort(sort); // Sort points using first sortCoordPts coordinates.
					else
						((CachedPointSet) p).sortByCoordinate(0); // Sort by first coordinate.
				}
				PointSetIterator stream = p.iterator();
				stream.resetCurPointIndex(); // Go to first point.
				int i = 0;
				for (T mc : chains) { // Assume the chains are sorted
					if (mc.hasStopped()) {
						numNotStopped--;
					} else {
						stream.setCurCoordIndex(sortCoordPts); // Skip first sortCoordPts coord.
						mc.nextStep(stream); // simulate next step of the chain.
						stream.resetNextSubstream(); // Go to next point.
						if (mc.hasStopped())
							numNotStopped--;
					}
					performances[i] = mc.getPerformance();
					++i;
				}
				++step;
			}
			// System.out.println("calcMeanPerf"+calcMeanPerf());
			return calcMeanPerf();

		}//goto

	}

	/**
	 * This version assumes that `sortCoordPts = 0`, so that there is no need to
	 * sort the points at each step.
	 */
	public double simulArrayRQMC(PointSet p, PointSetRandomization rand, MultiDimSort sort, double numSteps) {
		return simulArrayRQMC(p, rand, sort, 0, numSteps);
	}

	/**
	 * This version assumes that `sortCoordPts = 0` and uses the preset
	 * randomization and sort for the chains.
	 */
	public double simulArrayRQMC(PointSet p, double numSteps) {
		return simulArrayRQMC(p, randomization, savedSort, 0, numSteps);
	}

	/**
	 * Returns the vector for performances for the @f$n@f$ chains.
	 */
	public double[] getPerformances() {
		return performances;
	}

	/**
	 * Computes and returns the mean performance of the @f$n@f$ chains.
	 */
	public double calcMeanPerf() {
		double sumPerf = 0.0; // Sum of performances.
		for (int i = 0; i < n; ++i) {
			sumPerf += performances[i];
		}
		return sumPerf / n;
	}

	/**
	 * Performs <tt>m</tt> independent replications of an array-RQMC simulation as
	 * in <tt>simulArrayRQMC</tt>. The statistics on the <tt>m</tt> corresponding
	 * averages are collected in <tt>statReps</tt>.
	 */
	public void simulReplicatesArrayRQMC(PointSet p, PointSetRandomization rand, MultiDimSort sort, int sortCoordPts,
			double numSteps, int m, Tally statReps) {
		makeCopies(p.getNumPoints());
		statReps.init();
		for (int rep = 0; rep < m; rep++) {
			statReps.add(simulArrayRQMC(p, rand, sort, sortCoordPts, numSteps));
		}
	}

	/**
	 * Performs <tt>m</tt> independent replications of an array-RQMC simulation as
	 * in <tt>simulFormatArrayRQMC</tt>. The statistics on the <tt>m</tt>
	 * corresponding averages are collected in <tt>statReps</tt> and the results are
	 * also returned in a string.
	 */
	public String simulReplicatesArrayRQMCFormat(PointSet p, PointSetRandomization rand, MultiDimSort sort,
			int sortCoordPts, double numSteps, int m, Tally statReps) {
		Chrono timer = Chrono.createForSingleThread();
		makeCopies(p.getNumPoints());
		timer.init();
		statReps.init();
		for (int rep = 0; rep < m; rep++) {
			statReps.add(simulArrayRQMC(p, rand, sort, sortCoordPts, numSteps));
		}
		StringBuffer sb = new StringBuffer("----------------------------------------------" + PrintfFormat.NEWLINE);
		sb.append("Array-RQMC simulations:" + PrintfFormat.NEWLINE);
		sb.append(PrintfFormat.NEWLINE + p.toString() + ":" + PrintfFormat.NEWLINE);
		sb.append(" Number of indep copies m  = " + m);
		sb.append(PrintfFormat.NEWLINE + " Number of points n        = " + n + PrintfFormat.NEWLINE);
		sb.append(baseChain.formatResultsRQMC(statReps, n));
		sb.append(" CPU Time = " + timer.format() + PrintfFormat.NEWLINE);
		return sb.toString();
	}

	/**
	 * Returns a string that reports the the ratio of MC variance per run `varMC`
	 * over the RQMC variance per run `varRQMC`.
	 */
	public String varianceImprovementFormat(double varRQMC, double varMC) {
		// double varRQMC = p.getNumPoints() * statReps.variance();
		StringBuffer sb = new StringBuffer(
				" Variance ratio MC / RQMC: " + PrintfFormat.format(15, 10, 4, varMC / varRQMC) + PrintfFormat.NEWLINE);
		return sb.toString();
	}

	/**
	 * Performs an experiment to estimate the convergence rate of the RQMC variance
	 * as a function of @f$n@f$, by invoking #simulReplicatesArrayRQMC(r) epeatedly
	 * with a given array of point sets of different sizes @f$n@f$. Returns a string
	 * that reports the mean, variance, and variance reduction factor (VRF) with
	 * respect to MC for each point set, and the estimated convergence rate of the
	 * RQMC variance as a function of @f$n@f$. Assumes that <tt>varMC</tt> is the
	 * variance per run for MC. The string `methodLabel` should be a brief
	 * descriptor of the method (e.g., the type of point set, type of randomization,
	 * and type of sort). If the string `filenamePlot != null`, then the method also
	 * creates a `.tex` file with that name that contains a plot in log scale of the
	 * variance vs @f$n@f$.
	 */
	public String testVarianceRateFormat(PointSet[] pointSets, PointSetRandomization rand, MultiDimSort sort,
			int sortCoordPts, double numSteps, int m, double varMC, String filenamePlot, String methodLabel) {
		label = methodLabel;
		int numSets = pointSets.length; // Number of point sets.
		Tally statPerf = new Tally("Performance");
		double[] logn = new double[numSets];
		double[] variance = new double[numSets];
		double[] logVariance = new double[numSets];
		long initTime; // For timings.

		StringBuffer str = new StringBuffer("\n\n --------------------------");
		str.append(methodLabel + "\n  MC Variance : " + varMC + "\n\n");

		// Array-RQMC experiment with each pointSet.
		for (int i = 0; i < numSets; ++i) {
			initTime = System.currentTimeMillis();
			n = pointSets[i].getNumPoints();
			str.append("n = " + n + "\n");
			simulReplicatesArrayRQMC(pointSets[i], rand, sort, sortCoordPts, numSteps, m, statPerf);
			logn[i] = Num.log2(n);
			variance[i] = statPerf.variance();
			logVariance[i] = Num.log2(variance[i]);
			str.append("  Average = " + statPerf.average() + "\n");
			str.append(" RQMC Variance : " + n * variance[i] + "\n\n");
			str.append("  VRF =  " + varMC / (n * variance[i]) + "\n");
			str.append(formatTime((System.currentTimeMillis() - initTime) / 1000.) + "\n");
		}
		// Estimate regression slope and print plot and overall results.
		double regSlope = slope(logn, logVariance, numSets);
		str.append("Regression slope (log) for variance = " + regSlope + "\n\n");

		String[] tableField = { "log(n)", "log(Var)" };
		double[][] data = new double[numSets][2];
		for (int s = 0; s < numSets; s++) { // For each cardinality n
			data[s][0] = logn[s];
			data[s][1] = logVariance[s];
		}

		// Print plot and overall results in files.
		if (filenamePlot != null)
			try {
				/*
				 * Writer file = new FileWriter (filenamePlot + ".tex"); XYLineChart chart = new
				 * XYLineChart(); // ("title", "$log_2(n)$", "$log_2 Var[hat mu_{rqmc,s,n}]$");
				 * chart.add (logn, logVariance); file.write (chart.toLatex(12, 8));
				 * file.close();
				 */
				PgfDataTable pgf = new PgfDataTable(filenamePlot, pointSets.toString(), tableField, data);
				String pVar = pgf.drawPgfPlotSingleCurve(filenamePlot, "axis", 0, 1, 2, "", "");
				String plotIV = (PgfDataTable.pgfplotFileHeader() + pVar + PgfDataTable.pgfplotEndDocument());

				FileWriter fileIV = new FileWriter(filenamePlot + "_" + "VAr.tex");
				fileIV.write(plotIV);
				fileIV.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

		return str.toString();
	}

	public String testVarianceRateFormat(RQMCPointSet[] rqmcPts, MultiDimSort sort,
			int sortCoordPts, int numSteps, int m, double varMC, String filenamePlot, String methodLabel) {
		label = methodLabel;
		int numSets = rqmcPts.length; // Number of point sets.
		Tally statPerf = new Tally("Performance");
		double[] logn = new double[numSets];
		double[] variance = new double[numSets];
		double[] logVariance = new double[numSets];
		long initTime; // For timings.

		StringBuffer str = new StringBuffer("\n\n --------------------------");
		str.append(methodLabel + "\n  MC Variance : " + varMC + "\n\n");

		// Array-RQMC experiment with each pointSet.
		for (int i = 0; i < numSets; ++i) {
			initTime = System.currentTimeMillis();
			n = rqmcPts[i].getNumPoints();
			str.append("n = " + n + "\n");
			simulReplicatesArrayRQMC(rqmcPts[i].getPointSet(),rqmcPts[i].getRandomization(), sort, sortCoordPts, numSteps, m, statPerf);
			logn[i] = Num.log2(n);
			variance[i] = statPerf.variance();
			logVariance[i] = Num.log2(variance[i]);
			str.append("  Average = " + statPerf.average() + "\n");
			str.append(" RQMC Variance : " + n * variance[i] + "\n\n");
			str.append("  VRF =  " + varMC / (n * variance[i]) + "\n");
			str.append(formatTime((System.currentTimeMillis() - initTime) / 1000.) + "\n");
		}
		// Estimate regression slope and print plot and overall results.
		double regSlope = slope(logn, logVariance, numSets);
		str.append("Regression slope (log) for variance = " + regSlope + "\n\n");

		String[] tableField = { "log(n)", "log(Var)" };
		double[][] data = new double[numSets][2];
		for (int s = 0; s < numSets; s++) { // For each cardinality n
			data[s][0] = logn[s];
			data[s][1] = logVariance[s];
		}

		// Print plot and overall results in files.
		if (filenamePlot != null)
			try {
				/*
				 * Writer file = new FileWriter (filenamePlot + ".tex"); XYLineChart chart = new
				 * XYLineChart(); // ("title", "$log_2(n)$", "$log_2 Var[hat mu_{rqmc,s,n}]$");
				 * chart.add (logn, logVariance); file.write (chart.toLatex(12, 8));
				 * file.close();
				 */
				PgfDataTable pgf = new PgfDataTable(filenamePlot, rqmcPts[0].getLabel(), tableField, data);
				String pVar = pgf.drawPgfPlotSingleCurve(filenamePlot, "axis", 0, 1, 2, "", "");
				String plotIV = (PgfDataTable.pgfplotFileHeader() + pVar + PgfDataTable.pgfplotEndDocument());

				FileWriter fileIV = new FileWriter(filenamePlot + "_" + "VAr.tex");
				fileIV.write(plotIV);
				fileIV.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

		return str.toString();
	}
	/**
	 * Sorts the chains that have not stopped yet using the stored
	 * 
	 * @ref umontreal.ssj.util.MultiDimSort. All the stopped chains are placed at
	 *      the end, then the chains that have not stopped are sorted.
	 */
	public void sortNotStoppedChains(MultiDimSort sort) {
		int j = n - 1;
		int i = 0;
		T mc;
		while (j >= 0 && chains[j].hasStopped())
			--j;
		while (i < n && !chains[i].hasStopped())
			++i;
		while (i < j) {
			while (!chains[i].hasStopped())
				++i;
			while (chains[j].hasStopped())
				--j;
			mc = chains[i];
			chains[i] = chains[j];
			chains[j] = mc;
		}
		sort.sort(chains, 0, i);
	}

	/**
	 * Sorts the chains using the stored
	 * 
	 * @ref umontreal.ssj.util.MultiDimSort.
	 */
	public void sortChains() {
		savedSort.sort(chains, 0, n);
	}

	// Takes time in seconds and formats it.
	public String formatTime(double time) {
		int second, hour, min, centieme;
		hour = (int) (time / 3600.0);
		if (hour > 0) {
			time -= ((double) hour * 3600.0);
		}
		min = (int) (time / 60.0);
		if (min > 0) {
			time -= ((double) min * 60.0);
		}
		second = (int) time;
		centieme = (int) (100.0 * (time - (double) second) + 0.5);
		return String.valueOf(hour) + ":" + min + ":" + second + "." + centieme;
	}

	// Compute slope of linear regression of y on x, using only first n
	// observations.
	public double slope(double[] x, double[] y, int n) {
		if (n < 2) {
			return 0.0;
		} else {
			double[] x2 = new double[n], y2 = new double[n];
			for (int i = 0; i < n; ++i) {
				x2[i] = x[i];
				y2[i] = y[i];
			}
			return LeastSquares.calcCoefficients(x2, y2, 1)[1];
		}
	}
}
