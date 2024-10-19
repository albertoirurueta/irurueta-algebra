/*
 * Copyright (C) 2015 Alberto Irurueta Carro (alberto@irurueta.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.irurueta.statistics;

import com.irurueta.algebra.*;

import java.util.Arrays;

/**
 * Contains methods to work with multivariate normal (i.e. Gaussian)
 * distributions.
 */
public class MultivariateNormalDist {

    /**
     * Mean value of Gaussian distribution.
     */
    private double[] mu;

    /**
     * Covariance of Gaussian distribution.
     */
    private Matrix cov;

    /**
     * Basis in which the covariance matrix is expressed.
     * This value is obtained after decomposition.
     */
    private Matrix covBasis;

    /**
     * Variances on each direction of the basis.
     */
    private double[] variances;

    /**
     * Constructor.
     * Creates a normal distribution of 1 dimension.
     */
    public MultivariateNormalDist() {
        this(1);
    }

    /**
     * Constructor.
     * Creates a multivariate normal distribution having the provided
     * number of dimensions, with zero mean and unitary independent variances.
     *
     * @param dims number of dimensions. Must be greater than zero.
     * @throws IllegalArgumentException if provided number of dimensions is
     *                                  zero or less.
     */
    public MultivariateNormalDist(final int dims) {
        if (dims <= 0) {
            throw new IllegalArgumentException("number of dimensions must be greater than zero");
        }

        mu = new double[dims];
        try {
            cov = Matrix.identity(dims, dims);
        } catch (final WrongSizeException e) {
            throw new IllegalArgumentException("number of dimensions must be greater than zero", e);
        }
    }

    /**
     * Constructor.
     * Creates a multivariate normal distribution having provided mean and
     * covariance.
     *
     * @param mean       array containing mean. Must have the same number of rows as
     *                   provided covariance matrix
     * @param covariance matrix containing covariance. Must be square, symmetric
     *                   and positive definite (i.e. non-singular) and must have the same number
     *                   of rows as provided mean.
     * @throws IllegalArgumentException         if provided mean array has length
     *                                          smaller than 1 or if length of mean array is not the same as the number
     *                                          of rows of covariance matrix.
     * @throws InvalidCovarianceMatrixException if provided covariance matrix is
     *                                          not square, symmetric and positive definite (i.e. non singular).
     */
    public MultivariateNormalDist(final double[] mean, final Matrix covariance)
            throws InvalidCovarianceMatrixException {
        setMeanAndCovariance(mean, covariance);
    }

    /**
     * Constructor.
     * Creates a multivariate normal distribution having provided mean and
     * covariance.
     *
     * @param mean                              array containing mean. Must have the same number of rows as
     *                                          provided covariance matrix.
     * @param covariance                        matrix containing covariance. Must be square, symmetric
     *                                          and positive definite (i.e. non-singular) and must have the same number
     *                                          of rows as provided mean.
     * @param validateSymmetricPositiveDefinite true if covariance matrix must
     *                                          be validated to be positive definite, false to skip validation.
     * @throws IllegalArgumentException         if provided mean array has length
     *                                          smaller than 1 or if length of mean array is not the same as the number
     *                                          of rows of covariance matrix.
     * @throws InvalidCovarianceMatrixException if provided matrix is not
     *                                          valid (nor square or symmetric positive definite if validation is
     *                                          enabled).
     */
    public MultivariateNormalDist(
            final double[] mean, final Matrix covariance, final boolean validateSymmetricPositiveDefinite)
            throws InvalidCovarianceMatrixException {
        setMeanAndCovariance(mean, covariance, validateSymmetricPositiveDefinite);
    }

    /**
     * Gets array containing mean of this multivariate Gaussian distribution.
     *
     * @return mean of multivariate Gaussian distribution.
     */
    public double[] getMean() {
        return mu;
    }

    /**
     * Sets mean of this multivariate Gaussian distribution.
     * Length of provided mean must be equal to the number of rows of provided
     * covariance, otherwise instance won't be ready.
     *
     * @param mu mean of multivariate Gaussian distribution.
     * @throws IllegalArgumentException if provided array has a length smaller
     *                                  than 1.
     */
    public void setMean(final double[] mu) {
        if (mu.length == 0) {
            throw new IllegalArgumentException("length of mean array must be greater than zero");
        }
        this.mu = mu;
    }

    /**
     * Gets matrix containing covariance of this multivariate Gaussian
     * distribution.
     *
     * @return covariance of multivariate Gaussian distribution.
     */
    public Matrix getCovariance() {
        return new Matrix(cov);
    }

    /**
     * Gets matrix containing covariance of this multivariate Gaussian
     * distribution.
     *
     * @param result instance where covariance of multivariate Gaussian
     *               distribution will be stored.
     */
    public void getCovariance(final Matrix result) {
        cov.copyTo(result);
    }

    /**
     * Sets covariance of this multivariate Gaussian distribution.
     *
     * @param cov covariance of this multivariate Gaussian distribution.
     * @throws InvalidCovarianceMatrixException if provided matrix is not valid
     *                                          (not square or symmetric positive definite).
     */
    public void setCovariance(final Matrix cov) throws InvalidCovarianceMatrixException {
        setCovariance(cov, true);
    }

    /**
     * Sets covariance of this multivariate Gaussian distribution.
     *
     * @param cov                               covariance of this multivariate Gaussian distribution.
     * @param validateSymmetricPositiveDefinite true if matrix must be
     *                                          validated to be positive definite, false to skip validation.
     * @throws InvalidCovarianceMatrixException if provided matrix is not
     *                                          valid (nor square or symmetric positive definite if validation is
     *                                          enabled).
     */
    public void setCovariance(final Matrix cov, final boolean validateSymmetricPositiveDefinite)
            throws InvalidCovarianceMatrixException {
        if (cov.getRows() != cov.getColumns()) {
            throw new InvalidCovarianceMatrixException("covariance matrix must be square");
        }

        try {
            if (validateSymmetricPositiveDefinite) {
                final var decomposer = new CholeskyDecomposer(cov);
                decomposer.decompose();
                if (!decomposer.isSPD()) {
                    throw new InvalidCovarianceMatrixException(
                            "covariance matrix must be symmetric positive definite (non singular)");
                }
            }

            this.cov = new Matrix(cov);
            covBasis = null;
            variances = null;
        } catch (final AlgebraException e) {
            throw new InvalidCovarianceMatrixException("covariance matrix must be square", e);
        }
    }

    /**
     * Sets mean and covariance of this multivariate Gaussian distribution.
     *
     * @param mu  array containing mean. Must have the same number of rows as
     *            provided covariance matrix
     * @param cov matrix containing covariance. Must be square, symmetric
     *            and positive definite (i.e. non-singular) and must have the same number
     *            of rows as provided mean.
     * @throws IllegalArgumentException         if provided mean array has length
     *                                          smaller than 1 or if length of mean array is not the same as the number
     *                                          of rows of covariance matrix.
     * @throws InvalidCovarianceMatrixException if provided covariance matrix is
     *                                          not square, symmetric and positive definite (i.e. non singular).
     */
    public final void setMeanAndCovariance(final double[] mu, final Matrix cov)
            throws InvalidCovarianceMatrixException {
        setMeanAndCovariance(mu, cov, true);
    }

    /**
     * Sets mean and covariance of this multivariate Gaussian distribution.
     *
     * @param mu                                array containing mean. Must have the same number of rows as
     *                                          provided covariance matrix
     * @param cov                               matrix containing covariance. Must be square, symmetric
     *                                          and positive definite (i.e. non-singular) and must have the same number
     *                                          of rows as provided mean.
     * @param validateSymmetricPositiveDefinite true if matrix must be
     *                                          validated to be positive definite, false to skip validation.
     * @throws IllegalArgumentException         if provided mean array has length
     *                                          smaller than 1 or if length of mean array is not the same as the number
     *                                          of rows of covariance matrix.
     * @throws InvalidCovarianceMatrixException if provided covariance matrix is
     *                                          not square, symmetric and positive definite (i.e. non singular).
     */
    public final void setMeanAndCovariance(
            final double[] mu, final Matrix cov, final boolean validateSymmetricPositiveDefinite)
            throws InvalidCovarianceMatrixException {
        if (mu.length != cov.getRows()) {
            throw new IllegalArgumentException("mean array length must be equal to covariance number of rows");
        }

        setCovariance(cov, validateSymmetricPositiveDefinite);
        setMean(mu);
    }

    /**
     * Indicates whether provided matrix is a valid covariance matrix.
     * A valid covariance matrix must be square, symmetric and positive definite
     * (i.e. non-singular).
     *
     * @param cov matrix to be checked.
     * @return true if matrix is a valid covariance matrix, false otherwise.
     */
    public static boolean isValidCovariance(final Matrix cov) {
        if (cov.getRows() != cov.getColumns()) {
            return false;
        }

        try {
            final var decomposer = new CholeskyDecomposer(cov);
            decomposer.decompose();
            return decomposer.isSPD();
        } catch (final AlgebraException e) {
            return false;
        }
    }

    /**
     * Indicates whether this instance is ready for any computation, false
     * otherwise.
     *
     * @return true if instance is ready, false otherwise.
     */
    public boolean isReady() {
        return mu != null && cov != null &&
                mu.length == cov.getRows();
    }

    /**
     * Basis containing on each column the direction of each variance in the
     * multidimensional Gaussian distribution, which is obtained from provided
     * covariance matrix.
     * This value is available only after the p.d.f. has been evaluated.
     *
     * @return basis containing on each column the direction of each variance
     * in the multidimensional Gaussian distribution.
     */
    public Matrix getCovarianceBasis() {
        return covBasis;
    }

    /**
     * Array containing the amount of variance on each direction of the basis
     * of the covariance in the multidimensional Gaussian distribution.
     * This value is available only after the p.d.f. has been evaluated.
     *
     * @return variance on each direction of the basis of the covariance.
     */
    public double[] getVariances() {
        return variances;
    }

    /**
     * Evaluates the probability density function (p.d.f.) of a multivariate
     * Gaussian distribution having current mean and covariance at point x.
     *
     * @param x array containing coordinates where p.d.f. is evaluated.
     * @return evaluation of p.d.f.
     * @throws NotReadyException            if this instance is not ready (mean and
     *                                      covariance have not been provided or are not valid).
     * @throws IllegalArgumentException     if provided point length is not valid.
     * @throws DecomposerException          happens if covariance is numerically
     *                                      unstable (i.e. contains NaNs or very large numbers).
     * @throws RankDeficientMatrixException happens if covariance is singular.
     */
    public double p(final double[] x) throws NotReadyException, DecomposerException, RankDeficientMatrixException {
        if (!isReady()) {
            throw new NotReadyException();
        }

        final var k = x.length;
        if (k != mu.length) {
            throw new IllegalArgumentException("length of point must be equal to the length of mean");
        }

        var detCov = 0.0;
        try {
            detCov = Utils.det(cov);
        } catch (final WrongSizeException ignore) {
            // never thrown
        }

        final var factor = 1.0 / (Math.sqrt(Math.pow(2.0 * Math.PI, k) * detCov));
        return factor * Math.exp(-0.5 * squaredMahalanobisDistance(x));
    }

    /**
     * Evaluates the cumulative distribution function (c.d.f.) of a Gaussian
     * distribution having current mean and covariance values.
     * The c.d.f. is equivalent to the joint probability of the multivariate
     * Gaussian distribution of having a value less than x on each direction
     * of the basis of independent variances obtained from covariance matrix.
     * Because the c.d.f is a probability, it always returns values between 0.0
     * and 1.0.
     * NOTE: this method will resize provided basis instance if needed.
     *
     * @param x     point where c.d.f. is evaluated.
     * @param basis instance where is stored the basis of each direction of
     *              independent covariances, if provided.
     * @return evaluation of c.d.f.
     * @throws IllegalArgumentException if length of provided point is not equal
     *                                  to length of current mean.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      if covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public double cdf(final double[] x, final Matrix basis) throws NotReadyException, DecomposerException {
        if (!isReady()) {
            throw new NotReadyException();
        }

        final var k = x.length;
        if (k != mu.length) {
            throw new IllegalArgumentException("length of point must be equal to the length of mean");
        }

        var p = 1.0;
        try {
            processCovariance();

            if (basis != null) {
                basis.copyFrom(covBasis);
            }

            for (int i = 0; i < k; i++) {
                final var singleBasis = covBasis.getSubmatrixAsArray(0, i, k - 1, i);
                final var coordX = ArrayUtils.dotProduct(x, singleBasis);
                final var coordMu = ArrayUtils.dotProduct(mu, singleBasis);
                p *= NormalDist.cdf(coordX, coordMu, Math.sqrt(variances[i]));
            }

        } catch (final DecomposerException e) {
            throw e;
        } catch (final AlgebraException ignore) {
            // never thrown
        }

        return p;
    }

    /**
     * Evaluates the cumulative distribution function (c.d.f.) of a Gaussian
     * distribution having current mean and covariance values.
     * The c.d.f. is equivalent to the joint probability of the multivariate
     * Gaussian distribution of having a value less than x on each direction
     * of the basis of independent variances obtained from covariance matrix.
     * Because the c.d.f is a probability, it always returns values between 0.0
     * and 1.0.
     *
     * @param x point where c.d.f. is evaluated.
     * @return evaluation of c.d.f.
     * @throws IllegalArgumentException if length of provided point is not equal
     *                                  to length of current mean.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      if covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public double cdf(double[] x) throws NotReadyException, DecomposerException {
        return cdf(x, null);
    }

    /**
     * Computes the joint probability of all probabilities provided in the
     * array. The joint probability is computed by multiplying all components of
     * the array, assuming that all probabilities are independent.
     *
     * @param p array containing probabilities for each independent variance
     *          direction that can be obtained from provided covariance matrix.
     * @return joint probability.
     */
    public static double jointProbability(final double[] p) {
        var jointP = 1.0;
        for (final var aP : p) {
            jointP *= aP;
        }
        return jointP;
    }

    /**
     * Evaluates the inverse cumulative distribution function of a multivariate
     * Gaussian distribution for current mean and covariance values and provided
     * probability values for each dimension of the multivariate Gaussian
     * distribution.
     * NOTE: this method will resize provided basis instance if needed.
     *
     * @param p      array containing probability values to evaluate the inverse
     *               c.d.f. on each dimension. Values in the array must be between 0.0 and
     *               1.0.
     * @param result coordinates of the value x for which the c.d.f. has values
     *               p.
     * @param basis  instance where is stored the basis of each direction of
     *               independent covariances, if provided.
     * @throws IllegalArgumentException if length of probabilities is not equal
     *                                  to mean length, or if result and length of probabilities are not equal,
     *                                  or if provided probabilities are not between 0.0 and 1.0.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      if covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public void invcdf(final double[] p, final double[] result, final Matrix basis) throws NotReadyException,
            DecomposerException {
        if (!isReady()) {
            throw new NotReadyException("mean and covariance not provided or invalid");
        }

        final var k = p.length;
        if (k != mu.length) {
            throw new IllegalArgumentException(
                    "length of probabilities must be equal to the length of mean");
        }
        if (k != result.length) {
            throw new IllegalArgumentException("length of result must be equal to the length of mean");
        }

        try {
            processCovariance();

            if (basis != null) {
                basis.copyFrom(covBasis);
            }

            // initialize to mean
            System.arraycopy(mu, 0, result, 0, k);
            for (var i = 0; i < k; i++) {
                final var singleBasis = covBasis.getSubmatrixAsArray(0, i, k - 1, i);
                final double coord = NormalDist.invcdf(p[i], mu[i], Math.sqrt(variances[i])) - mu[i];
                // coord*singleBasis
                ArrayUtils.multiplyByScalar(singleBasis, coord, singleBasis);

                // result = mean + coord*singleBasis
                ArrayUtils.sum(result, singleBasis, result);
            }
        } catch (final DecomposerException e) {
            throw e;
        } catch (final AlgebraException ignore) {
            // never thrown
        }
    }

    /**
     * Evaluates the inverse cumulative distribution function of a multivariate
     * Gaussian distribution for current mean and covariance values and provided
     * probability values for each dimension of the multivariate Gaussian
     * distribution.
     * NOTE: this method will resize provided basis instance if needed.
     *
     * @param p     array containing probability values to evaluate the inverse
     *              c.d.f. on each dimension. Values in the array must be between 0.0 and
     *              1.0.
     * @param basis instance where is stored the basis of each direction of
     *              independent covariances, if provided.
     * @return a new array containing coordinates of the value x for which the
     * c.d.f. has values p.
     * @throws IllegalArgumentException if length of probabilities is not equal
     *                                  to mean length, or if provided probabilities are not between 0.0 and 1.0.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      if covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public double[] invcdf(final double[] p, final Matrix basis) throws NotReadyException, DecomposerException {
        if (mu == null) {
            throw new NotReadyException("mean not defined");
        }

        final var result = new double[mu.length];
        invcdf(p, result, basis);
        return result;
    }

    /**
     * Evaluates the inverse cumulative distribution function of a multivariate
     * Gaussian distribution for current mean and covariance values and provided
     * probability values for each dimension of the multivariate Gaussian
     * distribution.
     *
     * @param p      array containing probability values to evaluate the inverse
     *               c.d.f. on each dimension. Values in the array must be between 0.0 and
     *               1.0.
     * @param result coordinates of the value x for which the c.d.f. has values
     *               p.
     * @throws IllegalArgumentException if length of probabilities is not equal
     *                                  to mean length, or if result and length of probabilities are not equal,
     *                                  or if provided probabilities are not between 0.0 and 1.0.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      if covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public void invcdf(final double[] p, final double[] result) throws NotReadyException, DecomposerException {
        invcdf(p, result, null);
    }

    /**
     * Evaluates the inverse cumulative distribution function of a multivariate
     * Gaussian distribution for current mean and covariance values and provided
     * probability values for each dimension of the multivariate Gaussian
     * distribution.
     *
     * @param p array containing probability values to evaluate the inverse
     *          c.d.f. on each dimension. Values in the array must be between 0.0 and
     *          1.0.
     * @return coordinates of the value x for which the c.d.f. has values
     * * p.
     * @throws IllegalArgumentException if length of probabilities is not equal
     *                                  to mean length, or if result and length of probabilities are not equal,
     *                                  or if provided probabilities are not between 0.0 and 1.0.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      if covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public double[] invcdf(final double[] p) throws NotReadyException, DecomposerException {
        return invcdf(p, (Matrix) null);
    }

    /**
     * Evaluates the inverse cumulative distribution function of a multivariate
     * Gaussian distribution for current mean and covariance values and provided
     * probability value.
     * Obtained result coordinates are computed taking into account the basis
     * of independent variances computed from current covariance matrix.
     * NOTE: notice that the inverse cdf of a mutivariate Gaussian distribution
     * does not have a unique solution. This method simply returns one of the
     * possible solutions by assuming equal probabilities on each dimension.
     * NOTE: this method will resize provided basis instance if needed.
     *
     * @param p      probability value to evaluate the inverse c.d.f. at. This value
     *               must be between 0.0 and 1.0
     * @param result coordinates of the value x for which the c.d.f. has value
     *               p.
     * @param basis  instance where is stored the basis of each direction of
     *               independent covariances, if provided.
     * @throws IllegalArgumentException if provided probability value is not
     *                                  between 0.0 and 1.0, if length of provided result array is not equal
     *                                  to length of current mean.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      if covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public void invcdf(final double p, final double[] result, final Matrix basis)
            throws NotReadyException, DecomposerException {
        if (p <= 0.0 || p >= 1.0) {
            throw new IllegalArgumentException("probability value must be between 0.0 and 1.0");
        }

        if (!isReady()) {
            throw new NotReadyException("mean and covariance not provided or invalid");
        }

        final var k = result.length;
        if (k != mu.length) {
            throw new IllegalArgumentException("length of result must be equal to mean length");
        }

        final var probs = new double[k];
        Arrays.fill(probs, Math.pow(p, 1.0 / k));
        invcdf(probs, result, basis);
    }

    /**
     * Evaluates the inverse cumulative distribution function of a multivariate
     * Gaussian distribution for current mean and covariance values and provided
     * probability value.
     * Obtained result coordinates are computed taking into account the basis
     * of independent variances computed from current covariance matrix.
     * NOTE: notice that the inverse cdf of a mutivariate Gaussian distribution
     * does not have a unique solution. This method simply returns one of the
     * possible solutions by assuming equal probabilities on each dimension.
     * NOTE: this method will resize provided basis instance if needed.
     *
     * @param p     probability value to evaluate the inverse c.d.f. at. This value
     *              must be between 0.0 and 1.0
     * @param basis instance where is stored the basis of each direction of
     *              independent covariances, if provided.
     * @return a new array containing the coordinates of the value x for which
     * the c.d.f. has value p.
     * @throws IllegalArgumentException if provided probability value is not
     *                                  between 0.0 and 1.0.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      f covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public double[] invcdf(final double p, final Matrix basis) throws NotReadyException, DecomposerException {
        if (mu == null) {
            throw new NotReadyException("mean not defined");
        }

        final var result = new double[mu.length];
        invcdf(p, result, basis);
        return result;
    }

    /**
     * Evaluates the inverse cumulative distribution function of a multivariate
     * Gaussian distribution for current mean and covariance values and provided
     * probability value.
     * Obtained result coordinates are computed taking into account the basis
     * of independent variances computed from current covariance matrix.
     * NOTE: notice that the inverse cdf of a mutivariate Gaussian distribution
     * does not have a unique solution. This method simply returns one of the
     * possible solutions by assuming equal probabilities on each dimension.
     *
     * @param p      probability value to evaluate the inverse c.d.f. at. This value
     *               must be between 0.0 and 1.0
     * @param result coordinates of the value x for which the c.d.f. has value
     *               p.
     * @throws IllegalArgumentException if provided probability value is not
     *                                  between 0.0 and 1.0, if length of provided result array is not equal
     *                                  to length of current mean.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      f covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public void invcdf(final double p, final double[] result) throws NotReadyException, DecomposerException {
        invcdf(p, result, null);
    }

    /**
     * Evaluates the inverse cumulative distribution function of a multivariate
     * Gaussian distribution for current mean and covariance values and provided
     * probability value.
     * Obtained result coordinates are computed taking into account the basis
     * of independent variances computed from current covariance matrix.
     * NOTE: notice that the inverse cdf of a mutivariate Gaussian distribution
     * does not have a unique solution. This method simply returns one of the
     * possible solutions by assuming equal probabilities on each dimension.
     *
     * @param p probability value to evaluate the inverse c.d.f. at. This value
     *          must be between 0.0 and 1.0
     * @return a new array containing the coordinates of the value x for which
     * the c.d.f. has value p.
     * @throws IllegalArgumentException if provided probability value is not
     *                                  between 0.0 and 1.0.
     * @throws NotReadyException        if this instance is not ready (mean and
     *                                  covariance have not been provided or are not valid).
     * @throws DecomposerException      f covariance is numerically unstable (i.e.
     *                                  contains NaNs or very large numbers).
     */
    public double[] invcdf(final double p) throws NotReadyException, DecomposerException {
        return invcdf(p, (Matrix) null);
    }

    /**
     * Computes the Mahalanobis distance of provided multivariate pot x for
     * current mean and covariance values.
     *
     * @param x point where Mahalanobis distance is evaluated.
     * @return Mahalanobis distance of provided point respect to mean.
     * @throws DecomposerException          happens if covariance is numerically
     *                                      unstable (i.e. contains NaNs or very large numbers).
     * @throws RankDeficientMatrixException happens if covariance is singular.
     */
    public double mahalanobisDistance(final double[] x) throws DecomposerException, RankDeficientMatrixException {
        return Math.sqrt(squaredMahalanobisDistance(x));
    }

    /**
     * Computes the squared Mahalanobis distance of provided multivariate pot x
     * for current mean and covariance values.
     *
     * @param x point where Mahalanobis distance is evaluated.
     * @return Mahalanobis distance of provided point respect to mean.
     * @throws DecomposerException          happens if covariance is numerically
     *                                      unstable (i.e. contains NaNs or very large numbers).
     * @throws RankDeficientMatrixException happens if covariance is singular.
     */
    public double squaredMahalanobisDistance(final double[] x) throws DecomposerException,
            RankDeficientMatrixException {
        final var diff = ArrayUtils.subtractAndReturnNew(x, mu);
        final var diffMatrix = Matrix.newFromArray(diff, true);
        final var transDiffMatrix = diffMatrix.transposeAndReturnNew();

        try {
            final var invCov = Utils.inverse(cov);
            transDiffMatrix.multiply(invCov);
            transDiffMatrix.multiply(diffMatrix);

        } catch (final WrongSizeException ignore) {
            // never thrown
        }

        return transDiffMatrix.getElementAtIndex(0);
    }

    /**
     * Processes current covariance by decomposing it into a basis and its
     * corresponding variances if needed.
     *
     * @throws DecomposerException   happens if covariance is numerically
     *                               unstable (i.e. contains NaNs or very large numbers).
     * @throws NotReadyException     never thrown because decomposer will always be
     *                               ready.
     * @throws LockedException       never thrown because decomposer will never  be
     *                               locked.
     * @throws NotAvailableException never thrown because first a
     *                               DecomposerException will be thrown before attempting to get V or
     *                               singular values.
     */
    public void processCovariance() throws DecomposerException, NotReadyException, LockedException,
            NotAvailableException {
        if (cov == null) {
            throw new NotReadyException("covariance must be defined");
        }

        if (covBasis == null || variances == null) {
            final var decomposer = new SingularValueDecomposer(cov);
            decomposer.decompose();

            // because matrix is symmetric positive definite:
            // And matrices U and V are orthonormal
            // Cov = A'*A = (U*S*V')'*(U*S*V')=V*S*U'*U*S*V' = V*S^2*V',

            // where matrix S is diagonal, and contains the standard deviations
            // on each direction of the basis V, and hence S^2 is also diagonal but
            // containing variances on each direction.
            // The values of S^2 are the eigenvalues of Cov, and V are the
            // eigenvectors of Cov, hence covariance can be expressed as variances
            // on each direction of the basis V.

            // matrix containing eigenvectors (basis of directions)
            covBasis = decomposer.getV();

            // array containing the eigenvalues (variances on each direction)
            variances = decomposer.getSingularValues();
        }
    }

    /**
     * Evaluates the Jacobian and a multivariate function at a certain mean
     * point and computes the non-linear propagation of Gaussian uncertainty
     * through such function at such point.
     *
     * @param evaluator  interface to evaluate a multivariate function and its
     *                   Jacobian at a certain point.
     * @param mean       mean of original multivariate Gaussian distribution to be
     *                   propagated. Must have the length of the number of input variables of the
     *                   multivariate function to be evaluated.
     * @param covariance covariance of original Gaussian distribution to be
     *                   propagated. Must be symmetric positive definite having size NxN where N
     *                   is the length of provided mean.
     * @param result     instance where propagated multiavariate Gaussian
     *                   distribution will be stored.
     * @throws WrongSizeException               if evaluator returns an invalid number of
     *                                          variables (i.e. negative or zero).
     * @throws InvalidCovarianceMatrixException if provided covariance matrix is
     *                                          not valid (i.e. is not symmetric positive definite).
     * @see <a href="https://github.com/joansola/slamtb">propagateUncertainty.m at https://github.com/joansola/slamtb</a>
     */
    public static void propagate(
            final JacobianEvaluator evaluator, final double[] mean, final Matrix covariance,
            final MultivariateNormalDist result) throws WrongSizeException, InvalidCovarianceMatrixException {

        final var ndims = mean.length;
        final var nvars = evaluator.getNumberOfVariables();
        final var evaluation = new double[nvars];
        final var jacobian = new Matrix(nvars, ndims);
        evaluator.evaluate(mean, evaluation, jacobian);

        // [y, Y_x] = f(x)
        // Y = Y_x * X * Y_x'
        final var jacobianTrans = jacobian.transposeAndReturnNew();
        jacobian.multiply(covariance);
        jacobian.multiply(jacobianTrans);

        // ensure that new covariance is symmetric positive definite
        jacobian.symmetrize();

        result.setMean(evaluation);
        result.setCovariance(jacobian, false);
    }

    /**
     * Evaluates the Jacobian and a multivariate function at a certain mean
     * point and computes the non-linear propagation of Gaussian uncertainty
     * through such function at such point.
     *
     * @param evaluator  interface to evaluate a multivariate function and its
     *                   Jacobian at a certain point.
     * @param mean       mean of original multivariate Gaussian distribution to be
     *                   propagated. Must have the length of the number of input variables of the
     *                   multivariate function to be evaluated.
     * @param covariance covariance of original Gaussian distribution to be
     *                   propagated. Must be symmetric positive definite having size NxN where N
     *                   is the length of provided mean.
     * @return a new propagated multivariate Gaussian distribution.
     * @throws WrongSizeException               if evaluator returns an invalid number of
     *                                          variables (i.e. negative or zero).
     * @throws InvalidCovarianceMatrixException if provided covariance matrix is
     *                                          not valid (i.e. is not symmetric positive definite).
     * @see <a href="https://github.com/joansola/slamtb">propagateUncertainty.m at https://github.com/joansola/slamtb</a>
     */
    public static MultivariateNormalDist propagate(
            final JacobianEvaluator evaluator, final double[] mean, final Matrix covariance)
            throws WrongSizeException, InvalidCovarianceMatrixException {
        final var result = new MultivariateNormalDist();
        propagate(evaluator, mean, covariance, result);
        return result;
    }

    /**
     * Evaluates the Jacobian and a multivariate function at a certain mean
     * point and computes the non-linear propagation of Gaussian uncertainty
     * through such function at such point.
     *
     * @param evaluator interface to evaluate a multivariate function and its
     *                  Jacobian at a certain point.
     * @param dist      multivariate Gaussian distribution to be propagated.
     * @param result    instance where propagated multivariate Gaussian
     *                  distribution will be stored.
     * @throws WrongSizeException if evaluator returns an invalid number of
     *                            variables (i.e. negative or zero).
     * @see <a href="https://github.com/joansola/slamtb">propagateUncertainty.m at https://github.com/joansola/slamtb</a>
     */
    public static void propagate(final JacobianEvaluator evaluator, final MultivariateNormalDist dist,
                                 final MultivariateNormalDist result) throws WrongSizeException {
        try {
            propagate(evaluator, dist.getMean(), dist.getCovariance(), result);
        } catch (final InvalidCovarianceMatrixException ignore) {
            // never thrown
        }
    }

    /**
     * Evaluates the Jacobian and a multivariate function at a certain mean
     * point and computes the non-linear propagation of Gaussian uncertainty
     * through such function at such point.
     *
     * @param evaluator interface to evaluate a multivariate function and its
     *                  Jacobian at a certain point.
     * @param dist      multivariate Gaussian distribution to be propagated.
     * @return a new propagated multivariate Gaussian distribution.
     * @throws WrongSizeException if evaluator returns an invalid number of
     *                            variables (i.e. negative or zero).
     * @see <a href="https://github.com/joansola/slamtb">propagateUncertainty.m at https://github.com/joansola/slamtb</a>
     */
    public static MultivariateNormalDist propagate(
            final JacobianEvaluator evaluator, final MultivariateNormalDist dist) throws WrongSizeException {
        final var result = new MultivariateNormalDist();
        propagate(evaluator, dist, result);
        return result;
    }

    /**
     * Evaluates the Jacobian and a multivariate function at the mean point of
     * this distribution and computes the non-linear propagation of Gaussian
     * uncertainty through such function at such point.
     *
     * @param evaluator interface to evaluate a multivariate function and its
     *                  Jacobian at a certain point.
     * @param result    instance where propagated multivariate Gaussian
     *                  distribution will be stored.
     * @throws WrongSizeException if evaluator returns an invalid number of
     *                            variables (i.e. negative or zero).
     * @see <a href="https://github.com/joansola/slamtb">propagateUncertainty.m at https://github.com/joansola/slamtb</a>
     */
    public void propagateThisDistribution(final JacobianEvaluator evaluator, final MultivariateNormalDist result)
            throws WrongSizeException {
        propagate(evaluator, this, result);
    }

    /**
     * Evaluates the Jacobian and a multivariate function at the mean point of
     * this distribution and computes the non-linear propagation of Gaussian
     * uncertainty through such function at such point.
     *
     * @param evaluator interface to evaluate a multivariate function and its
     *                  Jacobian at a certain point.
     * @return a new propagated multivariate Gaussian distribution.
     * @throws WrongSizeException if evaluator returns an invalid number of
     *                            variables (i.e. negative or zero).
     * @see <a href="https://github.com/joansola/slamtb">propagateUncertainty.m at https://github.com/joansola/slamtb</a>
     */
    public MultivariateNormalDist propagateThisDistribution(final JacobianEvaluator evaluator)
            throws WrongSizeException {
        final var result = new MultivariateNormalDist();
        propagateThisDistribution(evaluator, result);
        return result;
    }

    /**
     * Interface to evaluate a multivariate function at multivariate point x to
     * obtain multivariate result y and its corresponding jacobian at point x.
     */
    public interface JacobianEvaluator {
        /**
         * Evaluates multivariate point
         *
         * @param x        array containing multivariate point where function is
         *                 evaluated.
         * @param y        result of evaluating multivariate point.
         * @param jacobian jacobian of multivariate function at point x.
         */
        void evaluate(final double[] x, final double[] y, final Matrix jacobian);

        /**
         * Number of variables in output of evaluated function. This is equal
         * to the length of the array y obtained as function evaluations.
         *
         * @return number of variables of the function.
         */
        int getNumberOfVariables();
    }
}
