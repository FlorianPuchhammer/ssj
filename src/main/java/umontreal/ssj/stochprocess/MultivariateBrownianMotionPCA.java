/*
 * Class:        MultivariateBrownianMotionPCA
 * Description:  
 * Environment:  Java
 * Software:     SSJ 
 * Copyright (C) 2001  Pierre L'Ecuyer and Universite de Montreal
 * Organization: DIRO, Universite de Montreal
 * @author       Jean-Sébastien Parent
 * @since        2008

 * SSJ is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (GPL) as published by the
 * Free Software Foundation, either version 3 of the License, or
 * any later version.

 * SSJ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * A copy of the GNU General Public License is available at
   <a href="http://www.gnu.org/licenses">GPL licence site</a>.
 */
package umontreal.ssj.stochprocess;
import umontreal.ssj.rng.*;
import umontreal.ssj.probdist.*;
import umontreal.ssj.randvar.*;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.*;
import cern.colt.matrix.linalg.*;
import cern.colt.matrix.doublealgo.*;

/**
 * A multivariate Brownian motion process @f$\{\mathbf{X}(t) : t \geq0 \}@f$
 * sampled entirely using the *principal component* decomposition (PCA), as
 * explained in @cite fGLA04a&thinsp;, page 92. We construct the same matrix
 * @f$\boldsymbol{\Sigma}@f$ as in  @ref MultivariateBrownianMotion and
 * decompose it as @f$\boldsymbol{\Sigma}= B B^{\mathsf{t}}@f$ via PCA. We
 * also compute the matrix @f$\mathbf{C}@f$ whose element @f$(i,j)@f$ is
 * @f$\mathrm{Cov}[B(t_{i+1}),B(t_{j+1})] = \min(t_{i+1},t_{j+1})@f$ and its
 * PCA decomposition @f$\mathbf{C}= \mathbf{A}\mathbf{A}^{\mathsf{t}}@f$, as
 * in class  @ref BrownianMotionPCA.
 *
 * <div class="SSJ-bigskip"></div><div class="SSJ-bigskip"></div>
 */
public class MultivariateBrownianMotionPCA extends MultivariateBrownianMotion {

    // protected double[][]  covX;        //  Matrice de covariance du vecteur des observ.
                                          // covX [i*c+k][j*c+l] = Cov[X_k(t_{i+1}),X_l(t_{j+1})].
                                          // -->  We will not store this one explicitly!
    protected DoubleMatrix2D  C;            // C[i,j] = \min(t_{i+1},t_{j+1}).
    protected DoubleMatrix2D  BC, sortedBC, copyBC;
    protected DoubleMatrix2D  PcovZ, PC;          // C = AA' (PCA decomposition).
    protected double[]    z,zz,zzz;           // vector of c*d standard normals.
    protected int[] eigenIndex;           // The (j+1)-th generated normal, j>=1, should be placed
                                          // at position vector eigenIndex[j] in vector z.
                                          // eigenIndex[0..cd-1] should be a permutation of 0,...,cd-1.
    protected boolean decompPCA;

    /**
     * Constructs a new `MultivariateBrownianMotionPCA` with parameters
     * @f$\boldsymbol{\mu}= \mathtt{mu}@f$, @f$\boldsymbol{\sigma}=
     * \mathtt{sigma}@f$, correlation matrix @f$\mathbf{R}_z =
     * \mathtt{corrZ}@f$, and initial value @f$\mathbf{X}(t_0) =
     * \mathtt{x0}@f$. The normal variates @f$Z_j@f$ in are generated by
     * inversion using the  @ref umontreal.ssj.rng.RandomStream `stream`.
     */
    public MultivariateBrownianMotionPCA (int c, double[] x0, double[] mu,
                                          double[] sigma, double[][] corrZ,
                                          RandomStream stream) {
        this.gen = new NormalGen (stream, new NormalDist());
        setParams(c, x0, mu, sigma, corrZ);
    }

    /**
     * Constructs a new `MultivariateBrownianMotionPCA` with parameters
     * @f$\boldsymbol{\mu}= \mathtt{mu}@f$, @f$\boldsymbol{\sigma}=
     * \mathtt{sigma}@f$, correlation matrix @f$\mathbf{R}_z =
     * \mathtt{corrZ}@f$, and initial value @f$\mathbf{X}(t_0) =
     * \mathtt{x0}@f$. The normal variates @f$Z_j@f$ in are generated by
     * `gen`.
     */
    public MultivariateBrownianMotionPCA (int c, double[] x0, double[] mu,
                                          double[] sigma, double[][] corrZ,
                                          NormalGen gen) {
        this.gen = gen;
        setParams(c, x0, mu, sigma, corrZ);
    }


    public void setParams (int c, double[] x0, double[] mu, double[] sigma,
                           double[][] corrZ) {
        decompPCA = false;
        super.setParams(c, x0, mu, sigma, corrZ);
    }
public double[] generatePath(){
        double[] u = new double[c*d];
        for(int i = 0; i < c*d; i++)
            u[i] = gen.nextDouble();
        return generatePath(u);
    }

/**
 * Sets the parameters
 */
public double[] generatePath(double[] uniform01) {
       double sum;
       int i, j, k;
        if(!decompPCA){init();}
        for (j = 0; j < c*d; j++) // the first components are multiplied by the biggest eigenvalues
            z[j] = uniform01[j] * BC.getQuick(j, 0);
//         System.out.println("PCA");
        // now we need to permute the vector z in the right order
        for(j = 0; j < c*d; j++){
            zz[j] = z[(int)BC.getQuick(j, 1)];
        }
        for (j = 0; j < d; j++) {
           for (i = 0; i < c; i++) {
               sum = 0.0;
               for (k = 0; k < c; k++)
                   sum += PcovZ.getQuick(i,k) * zz[j*c+k];
               zzz[j*c+i] = sum;  // chaque bloc de z multiplié par \Sigma (i.e. B)
           }
       }
       // multiplication of zz (zz = Id(\Sigma) * z) by the matrix \tilde C (\tilde C is the matrix C with all its components multiplied by the Identity matrix of size c x c to obtain a matrix of size cd x cd.

       for (j = 0; j < d; j++) {
           for (i = 0; i < c; i++) {
               sum = 0.0;
               for (k = 0; k < d; k++){
               sum += PC.getQuick(j, k)*zzz[c*k+i];
               }
               path[(j+1)*c+i] = sum + mu[i] * (t[j+1] - t[0]) + x0[i];
           }
       }

       observationIndex = observationCounter = d;
       return path;
    }


   protected DoubleMatrix2D decompPCA (DoubleMatrix2D Sigma,
                                       double[] eigenValues)  {
      // L'objet SingularValueDecomposition permet de recuperer la matrice
      // des valeurs propres en ordre decroissant et celle des vecteurs propres de
      // sigma (pour une matrice symetrique et definie-positive seulement).
      SingularValueDecomposition sv = new SingularValueDecomposition(Sigma);
      DoubleMatrix2D D = sv.getS();    // diagonal
      // Calculer la racine carree des valeurs propres
        for (int i = 0; i < D.rows(); i++){
            D.setQuick (i, i, Math.sqrt(D.getQuick (i, i)));
            eigenValues[i] = D.getQuick(i,i);
        }
      DoubleMatrix2D P = sv.getV();   // right factor matrix
        return P; // returns the matrix of eigenvectors associate with the eigenValues
                  // in the vector eigenValue.  The values in eigenValue are the square root
                  // of the eigenvalues
   }


    protected void init() {
        super.init();
        int i, j;
        z = new double[c*d];
        zz = new double[c*d];
        zzz = new double[c*d];
        double[] lambdaC = new double[d];
        double[] etaB = new double[c];
// BC stocks the product of eigenvalues in column 0 and the associated rank in the products in column 1
        BC = new DenseDoubleMatrix2D(c*d, 2);
// sortedBC stocks the product of eigenvalues in decreasing order in column 0 and their initial rank in the product in column 1, so column 1 becomes the eigenIndex.
        sortedBC = new DenseDoubleMatrix2D(c*d, 2);
        PC = new DenseDoubleMatrix2D(d,d);    //the matrix P (unit eigenVectors) of PCA decomp for matrix C
        PcovZ = new DenseDoubleMatrix2D(c,c); //the matrix P (unit eigenVectors) of PCA decomp for matrix covZ

        // Initialize C, based on the observation times.
        C = new DenseDoubleMatrix2D(d, d);
        for (j = 0; j < d; j++) {
           for (i = j; i < d; i++) {
              C.setQuick(j, i, t[j+1]);
              C.setQuick(i, j, t[j+1]);
           }
        }

        PC = decompPCA(C, lambdaC);     // v_j and \lambda_j
        PcovZ = decompPCA(covZ, etaB);  // w_i \eta_i

        for(j = 0; j < d; j++)
            for(i = 0; i < c; i++){
                BC.setQuick( c * j + i, 0, lambdaC[j] * etaB[i] );
                BC.setQuick( c * j + i, 1,(double)(c * j + i) );
            }
        sortedBC = Sorting.quickSort.sort(BC, 0); // sort in ascending order...  we need to reverse to
                                                  // have descending order
        // the matrix returned by sort (sortedBC) is still linked to BC so we need a deep copy in order to be able to inverse the order
        DoubleMatrix2D copyBC = sortedBC.copy();

        for(i = 0; i < c*d; i++){
            BC.setQuick(i, 0, copyBC.getQuick(c*d-1-i, 0));  // to obtain the eigenvalues in increasing order
            BC.setQuick(i, 1, copyBC.getQuick(c*d-1-i, 1));  // reverse the index...
        }
        decompPCA = true;
    }
}