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

import com.irurueta.algebra.AlgebraException;
import com.irurueta.algebra.ArrayUtils;
import com.irurueta.algebra.DecomposerHelper;
import com.irurueta.algebra.Matrix;
import com.irurueta.algebra.NotReadyException;
import com.irurueta.algebra.Utils;
import com.irurueta.algebra.WrongSizeException;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class MultivariateNormalDistTest {

    private static final double MIN_RANDOM_VALUE = -100.0;
    private static final double MAX_RANDOM_VALUE = 100.0;

    private static final double ABSOLUTE_ERROR = 1e-6;
    private static final double LARGE_ABSOLUTE_ERROR = 1e-3;

    private static final int N_SAMPLES = 1000000;
    private static final double RELATIVE_ERROR = 0.05;

    @Test
    void testConstructor() throws AlgebraException, InvalidCovarianceMatrixException {
        // empty constructor
        MultivariateNormalDist dist = new MultivariateNormalDist();

        // check correctness
        assertArrayEquals(new double[1], dist.getMean(), 0.0);
        assertEquals(Matrix.identity(1, 1), dist.getCovariance());
        assertTrue(dist.isReady());
        assertNull(dist.getCovarianceBasis());
        assertNull(dist.getVariances());

        // constructor with dimensions
        dist = new MultivariateNormalDist(2);

        // check correctness
        assertArrayEquals(new double[2], dist.getMean(), 0.0);
        assertEquals(Matrix.identity(2, 2), dist.getCovariance());
        assertTrue(dist.isReady());
        assertNull(dist.getCovarianceBasis());
        assertNull(dist.getVariances());

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> new MultivariateNormalDist(0));

        // constructor with mean and covariance
        final var randomizer = new UniformRandomizer();
        final var mean = new double[2];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        dist = new MultivariateNormalDist(mean, cov);

        // check correctness
        assertArrayEquals(mean, dist.getMean(), 0.0);
        assertEquals(cov, dist.getCovariance());
        assertTrue(dist.isReady());
        assertNull(dist.getCovarianceBasis());
        assertNull(dist.getVariances());

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> new MultivariateNormalDist(new double[0], cov));
        assertThrows(IllegalArgumentException.class, () -> new MultivariateNormalDist(new double[3], cov));

        final var wrong = DecomposerHelper.getLeftLowerTriangulatorFactor(2);
        final var wrong2 = DecomposerHelper.getSingularMatrixInstance(2, 2);
        final var wrong3 = new Matrix(2, 3);
        assertThrows(InvalidCovarianceMatrixException.class, () -> new MultivariateNormalDist(mean, wrong));
        assertThrows(InvalidCovarianceMatrixException.class, () -> new MultivariateNormalDist(mean, wrong2));
        assertThrows(InvalidCovarianceMatrixException.class, () -> new MultivariateNormalDist(mean, wrong3));

        dist = new MultivariateNormalDist(mean, cov, false);

        // check correctness
        assertArrayEquals(mean, dist.getMean(), 0.0);
        assertEquals(cov, dist.getCovariance());
        assertTrue(dist.isReady());
        assertNull(dist.getCovarianceBasis());
        assertNull(dist.getVariances());
    }

    @Test
    void testGetSetMean() {
        final var dist = new MultivariateNormalDist();

        // check default value
        assertArrayEquals(new double[1], dist.getMean(), 0.0);

        // set new value
        final var mean = new double[2];
        final var randomizer = new UniformRandomizer();
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        dist.setMean(mean);

        // check correctness
        assertArrayEquals(mean, dist.getMean(), 0.0);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.setMean(new double[0]));
    }

    @Test
    void testGetSetCovariance() throws AlgebraException, InvalidCovarianceMatrixException {
        final var dist = new MultivariateNormalDist();

        // check default value
        assertEquals(dist.getCovariance(), Matrix.identity(1, 1));

        // set new value
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        dist.setCovariance(cov);

        // check correctness
        assertEquals(cov, dist.getCovariance());

        final var cov2 = new Matrix(2, 2);
        dist.getCovariance(cov2);
        assertEquals(cov, cov2);

        dist.setCovariance(cov, false);

        // check correctness
        assertEquals(cov, dist.getCovariance());

        // Force InvalidCovarianceMatrixException
        final var wrong = DecomposerHelper.getLeftLowerTriangulatorFactor(2);
        final var wrong2 = DecomposerHelper.getSingularMatrixInstance(2, 2);
        final var wrong3 = new Matrix(3, 2);
        assertThrows(InvalidCovarianceMatrixException.class, () -> dist.setCovariance(wrong));
        assertThrows(InvalidCovarianceMatrixException.class, () -> dist.setCovariance(wrong2));
        assertThrows(InvalidCovarianceMatrixException.class, () -> dist.setCovariance(wrong3));
    }

    @Test
    void testSetMeanAndCovariance() throws AlgebraException, InvalidCovarianceMatrixException {
        final var dist = new MultivariateNormalDist();

        // check default values
        assertArrayEquals(new double[1], dist.getMean(), 0.0);
        assertEquals(dist.getCovariance(), Matrix.identity(1, 1));

        // set new values
        final var randomizer = new UniformRandomizer();
        final var mean = new double[2];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        dist.setMeanAndCovariance(mean, cov);

        // check correctness
        assertArrayEquals(mean, dist.getMean(), 0.0);
        assertEquals(cov, dist.getCovariance());

        dist.setMeanAndCovariance(mean, cov, false);

        // check correctness
        assertArrayEquals(mean, dist.getMean(), 0.0);
        assertEquals(cov, dist.getCovariance());

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.setMeanAndCovariance(new double[0], cov));
        assertThrows(IllegalArgumentException.class, () -> dist.setMeanAndCovariance(new double[3], cov));

        final var wrong = DecomposerHelper.getLeftLowerTriangulatorFactor(2);
        final var wrong2 = DecomposerHelper.getSingularMatrixInstance(2, 2);
        final var wrong3 = new Matrix(2, 3);
        assertThrows(InvalidCovarianceMatrixException.class, () -> dist.setMeanAndCovariance(mean, wrong));
        assertThrows(InvalidCovarianceMatrixException.class, () -> dist.setMeanAndCovariance(mean, wrong2));
        assertThrows(InvalidCovarianceMatrixException.class, () -> dist.setMeanAndCovariance(mean, wrong3));
    }

    @Test
    void testIsValidCovariance() throws AlgebraException {
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        final var wrong = DecomposerHelper.getLeftLowerTriangulatorFactor(2);
        final var wrong2 = DecomposerHelper.getSingularMatrixInstance(2, 2);
        final var wrong3 = new Matrix(2, 3);

        assertTrue(MultivariateNormalDist.isValidCovariance(cov));
        assertFalse(MultivariateNormalDist.isValidCovariance(wrong));
        assertFalse(MultivariateNormalDist.isValidCovariance(wrong2));
        assertFalse(MultivariateNormalDist.isValidCovariance(wrong3));
    }

    @Test
    void testIsReady() {
        final var dist = new MultivariateNormalDist();

        // check initial value
        assertTrue(dist.isReady());

        // force not ready
        dist.setMean(new double[2]);
        assertFalse(dist.isReady());
    }

    @Test
    void testP() throws AlgebraException, InvalidCovarianceMatrixException {
        final var randomizer = new UniformRandomizer();
        final var mean = new double[2];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var x = new double[2];
        randomizer.fill(x, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        final var dist = new MultivariateNormalDist(mean, cov);

        assertEquals( 1.0 / (Math.sqrt(Math.pow(2.0 * Math.PI, 2.0) * Utils.det(cov)))
                        * Math.exp(-0.5 * ((Matrix.newFromArray(x).subtractAndReturnNew(Matrix.newFromArray(mean)))
                        .transposeAndReturnNew().multiplyAndReturnNew(Utils.inverse(cov)).multiplyAndReturnNew(
                                Matrix.newFromArray(x).subtractAndReturnNew(Matrix.newFromArray(mean))))
                        .getElementAtIndex(0)), dist.p(x), ABSOLUTE_ERROR);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.p(new double[1]));

        // Force NotReadyException
        dist.setMean(new double[1]);
        assertThrows(NotReadyException.class, () -> dist.p(x));
    }

    @Test
    void testCdf() throws AlgebraException, InvalidCovarianceMatrixException {
        final var randomizer = new UniformRandomizer();
        final var mean = new double[2];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        final var dist = new MultivariateNormalDist(mean, cov);

        // check that for 2 dimensions
        assertEquals(0.25, dist.cdf(mean), ABSOLUTE_ERROR);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.cdf(new double[1]));

        final var basis = new Matrix(2, 2);
        assertEquals(0.25, dist.cdf(mean, basis), ABSOLUTE_ERROR);
        assertEquals(basis, dist.getCovarianceBasis());

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.cdf(new double[3], basis));

        final var variances = dist.getVariances();
        assertNotNull(variances);

        final var v1 = basis.getSubmatrixAsArray(0, 0, 1, 0);
        final var v2 = basis.getSubmatrixAsArray(0, 1, 1, 1);

        // check in basis v1

        // -3 std away from mean on basis v1
        var x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                -3.0 * Math.sqrt(variances[0])));
        assertEquals(0.00135 * 0.5, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // -2 std away from mean on basis v1
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                -2.0 * Math.sqrt(variances[0])));
        assertEquals(0.02275 * 0.5, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // -1 std away from mean on basis v1
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1, -Math.sqrt(variances[0])));
        assertEquals(0.15866 * 0.5, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // on mean value
        x = mean;
        assertEquals(0.5 * 0.5, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // +1 std away from mean on basis v1
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1, Math.sqrt(variances[0])));
        assertEquals(0.84134 * 0.5, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // +2 std away from mean on basis v1
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                2.0 * Math.sqrt(variances[0])));
        assertEquals(0.97725 * 0.5, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // +3 std away from mean on basis v1
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                3.0 * Math.sqrt(variances[0])));
        assertEquals(0.99865 * 0.5, dist.cdf(x), LARGE_ABSOLUTE_ERROR);


        // check in basis v2

        // -3 std away from mean on basis v2
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v2,
                -3.0 * Math.sqrt(variances[1])));
        assertEquals(0.5 * 0.00135, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // -2 std away from mean on basis v2
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v2,
                -2.0 * Math.sqrt(variances[1])));
        assertEquals(0.5 * 0.02275, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // -1 std away from mean on basis v2
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v2, -Math.sqrt(variances[1])));
        assertEquals(0.5 * 0.15866, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // on mean value
        x = mean;
        assertEquals(0.5 * 0.5, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // +1 std away from mean on basis v2
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v2, Math.sqrt(variances[1])));
        assertEquals(0.5 * 0.84134, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // +2 std away from mean on basis v2
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v2,
                2.0 * Math.sqrt(variances[1])));
        assertEquals(0.5 * 0.97725, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // +3 std away from mean on basis v2
        x = ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v2,
                3.0 * Math.sqrt(variances[1])));
        assertEquals(0.5 * 0.99865, dist.cdf(x), LARGE_ABSOLUTE_ERROR);


        // check in both basis

        // -3 std away from mean on basis v1 and v2
        x = ArrayUtils.sumAndReturnNew(ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                        -3.0 * Math.sqrt(variances[0]))), ArrayUtils.multiplyByScalarAndReturnNew(v2,
                -3.0 * Math.sqrt(variances[1])));
        assertEquals(0.00135 * 0.00135, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // -2 std away from mean on basis v1 and v2
        x = ArrayUtils.sumAndReturnNew(ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                        -2.0 * Math.sqrt(variances[0]))), ArrayUtils.multiplyByScalarAndReturnNew(v2,
                -2.0 * Math.sqrt(variances[1])));
        assertEquals(0.02275 * 0.02275, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // -1 std away from mean on basis v1 and v2
        x = ArrayUtils.sumAndReturnNew(ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                        -Math.sqrt(variances[0]))), ArrayUtils.multiplyByScalarAndReturnNew(v2,
                -Math.sqrt(variances[1])));
        assertEquals(0.15866 * 0.15866, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // on mean value
        x = mean;
        assertEquals(0.5 * 0.5, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // +1 std away from mean on basis v1 and v2
        x = ArrayUtils.sumAndReturnNew(ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                        Math.sqrt(variances[0]))), ArrayUtils.multiplyByScalarAndReturnNew(v2,
                Math.sqrt(variances[1])));
        assertEquals(0.84134 * 0.84134, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // +2 std away from mean on basis v1 and v2
        x = ArrayUtils.sumAndReturnNew(ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                        2.0 * Math.sqrt(variances[0]))), ArrayUtils.multiplyByScalarAndReturnNew(v2,
                2.0 * Math.sqrt(variances[1])));
        assertEquals(0.97725 * 0.97725, dist.cdf(x), LARGE_ABSOLUTE_ERROR);

        // +3 std away from mean on basis v1 and v2
        x = ArrayUtils.sumAndReturnNew(ArrayUtils.sumAndReturnNew(mean, ArrayUtils.multiplyByScalarAndReturnNew(v1,
                        3.0 * Math.sqrt(variances[0]))), ArrayUtils.multiplyByScalarAndReturnNew(v2,
                3.0 * Math.sqrt(variances[1])));
        assertEquals(0.99865 * 0.99865, dist.cdf(x), LARGE_ABSOLUTE_ERROR);


        // Force NotReadyException
        final var x2 = ArrayUtils.sumAndReturnNew(ArrayUtils.sumAndReturnNew(mean,
                ArrayUtils.multiplyByScalarAndReturnNew(v1, 3.0 * Math.sqrt(variances[0]))),
                ArrayUtils.multiplyByScalarAndReturnNew(v2, 3.0 * Math.sqrt(variances[1])));
        dist.setMean(new double[1]);
        assertThrows(NotReadyException.class, () -> dist.cdf(x2));
        assertThrows(NotReadyException.class, () -> dist.cdf(x2, basis));
    }

    @Test
    void testJointProbability() {
        final var randomizer = new UniformRandomizer();
        final var p = new double[2];
        randomizer.fill(p);

        final var jointProbability = MultivariateNormalDist.jointProbability(p);

        assertTrue(jointProbability >= 0.0 && jointProbability <= 1.0);
        assertEquals(p[0] * p[1], jointProbability, 0.0);
    }

    @Test
    void testInvcdf() throws AlgebraException, InvalidCovarianceMatrixException {
        final var randomizer = new UniformRandomizer();
        final var mean = new double[2];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        final var p = new double[2];
        randomizer.fill(p); //values between 0.0 and 1.0

        final var dist = new MultivariateNormalDist(mean, cov);

        assertEquals(dist.cdf(dist.invcdf(p)), MultivariateNormalDist.jointProbability(p), ABSOLUTE_ERROR);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(new double[1]));

        final var basis = new Matrix(2, 2);
        assertEquals(dist.cdf(dist.invcdf(p, basis)), MultivariateNormalDist.jointProbability(p), ABSOLUTE_ERROR);
        assertEquals(dist.getCovarianceBasis(), basis);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(new double[1], basis));

        final var result = new double[2];
        dist.invcdf(p, result);
        assertEquals(dist.cdf(result), MultivariateNormalDist.jointProbability(p), ABSOLUTE_ERROR);
        assertEquals(dist.getCovarianceBasis(), basis);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(new double[1], result));
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(p, new double[1]));

        dist.invcdf(p, result, basis);
        assertEquals(dist.cdf(result), MultivariateNormalDist.jointProbability(p), ABSOLUTE_ERROR);
        assertEquals(dist.getCovarianceBasis(), basis);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(new double[1], result, basis));
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(p, new double[1], basis));

        // Force NotReadyException
        dist.setMean(new double[1]);
        assertThrows(NotReadyException.class, () -> dist.invcdf(p));
        assertThrows(NotReadyException.class, () -> dist.invcdf(p, basis));
        assertThrows(NotReadyException.class, () -> dist.invcdf(p, result));
        assertThrows(NotReadyException.class, () -> dist.invcdf(p, result, basis));
    }

    @Test
    void testInvcdfJointProbability() throws AlgebraException, InvalidCovarianceMatrixException {
        final var randomizer = new UniformRandomizer();
        final var mean = new double[2];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        final var jointP = randomizer.nextDouble();
        final var singleP = Math.sqrt(jointP);
        final var p = new double[]{singleP, singleP};

        final var dist = new MultivariateNormalDist(mean, cov);

        final var x = dist.invcdf(p);

        assertArrayEquals(x, dist.invcdf(dist.cdf(x)), ABSOLUTE_ERROR);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(0.0));
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(1.0));

        final var basis = new Matrix(2, 2);
        assertArrayEquals(x, dist.invcdf(dist.cdf(x), basis), ABSOLUTE_ERROR);
        assertEquals(dist.getCovarianceBasis(), basis);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(0.0, basis));
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(1.0, basis));

        final var result = new double[2];
        dist.invcdf(dist.cdf(x), result);
        assertArrayEquals(x, result, ABSOLUTE_ERROR);
        assertEquals(dist.getCovarianceBasis(), basis);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(dist.cdf(x), new double[1]));

        final var cdf = dist.cdf(x);
        dist.invcdf(cdf, result, basis);
        assertArrayEquals(x, result, ABSOLUTE_ERROR);
        assertEquals(dist.getCovarianceBasis(), basis);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> dist.invcdf(dist.cdf(x), new double[1], basis));

        // Force NotReadyException
        dist.setMean(new double[1]);
        assertThrows(NotReadyException.class, () -> dist.invcdf(cdf));
        assertThrows(NotReadyException.class, () -> dist.invcdf(cdf, basis));
        assertThrows(NotReadyException.class, () -> dist.invcdf(cdf, result));
        assertThrows(NotReadyException.class, () -> dist.invcdf(cdf, result, basis));
    }

    @Test
    void testMahalanobisDistance() throws AlgebraException, InvalidCovarianceMatrixException {
        // check for 2 dimensions
        final var randomizer = new UniformRandomizer();
        var mean = new double[2];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        var x = new double[2];
        randomizer.fill(x, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        var dist = new MultivariateNormalDist(mean, cov);

        final var meanMatrix = Matrix.newFromArray(mean);
        final var xMatrix = Matrix.newFromArray(x);
        final var diffMatrix = xMatrix.subtractAndReturnNew(meanMatrix);
        final var transDiffMatrix = diffMatrix.transposeAndReturnNew();
        final var invCov = Utils.inverse(cov);

        final var value = transDiffMatrix.multiplyAndReturnNew(invCov).multiplyAndReturnNew(diffMatrix);
        assertEquals(1, value.getRows());
        assertEquals(1, value.getColumns());

        assertEquals(value.getElementAtIndex(0), dist.squaredMahalanobisDistance(x), ABSOLUTE_ERROR);
        assertEquals(Math.sqrt(value.getElementAt(0, 0)), dist.mahalanobisDistance(x), ABSOLUTE_ERROR);

        // check for 1 dimension
        mean = new double[1];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        x = new double[1];
        randomizer.fill(x, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        cov = DecomposerHelper.getSymmetricMatrix(1);

        dist = new MultivariateNormalDist(mean, cov);

        assertEquals(dist.squaredMahalanobisDistance(x), Math.pow(NormalDist.mahalanobisDistance(x[0], mean[0],
                Math.sqrt(cov.getElementAtIndex(0))), 2.0), ABSOLUTE_ERROR);
        assertEquals(dist.mahalanobisDistance(x), NormalDist.mahalanobisDistance(x[0], mean[0],
                Math.sqrt(cov.getElementAtIndex(0))), ABSOLUTE_ERROR);
    }

    @Test
    void testProcessCovariance() throws AlgebraException, InvalidCovarianceMatrixException {
        final var randomizer = new UniformRandomizer();
        final var mean = new double[2];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var x = new double[2];
        randomizer.fill(x, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));

        final var dist = new MultivariateNormalDist(mean, cov);

        // check default values
        assertNull(dist.getCovarianceBasis());
        assertNull(dist.getVariances());

        // process
        dist.processCovariance();

        // check correctness
        assertNotNull(dist.getCovarianceBasis());
        assertNotNull(dist.getVariances());

        // check that basis is orthonormal (its transpose is its inverse)
        final var basis = dist.getCovarianceBasis();
        assertTrue(Matrix.identity(2, 2).equals(basis.multiplyAndReturnNew(basis.transposeAndReturnNew()),
                ABSOLUTE_ERROR));

        // check that covariance can be expressed as: basis * variances * basis'
        final var variances = dist.getVariances();
        final var cov2 = new Matrix(2, 2);
        cov2.copyFrom(basis);
        cov2.multiply(Matrix.diagonal(variances));
        cov2.multiply(basis.transposeAndReturnNew());

        assertTrue(cov.equals(cov2, ABSOLUTE_ERROR));
    }

    @Test
    void testPropagate() throws WrongSizeException, InvalidCovarianceMatrixException {
        final var randomizer = new UniformRandomizer();
        final var mean = new double[2];
        randomizer.fill(mean, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var cov = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(2));
        cov.multiplyByScalar(1e-4);

        final var dist = new MultivariateNormalDist(mean, cov);
        final var evaluator = new MultivariateNormalDist.JacobianEvaluator() {
                    @Override
                    public void evaluate(final double[] x, final double[] y, final Matrix jacobian) {
                        y[0] = x[0] * x[0] * x[1];
                        y[1] = 5 * x[0] + Math.sin(x[1]);

                        jacobian.setElementAt(0, 0, 2 * x[0] * x[1]);
                        jacobian.setElementAt(0, 1, x[0] * x[0]);

                        jacobian.setElementAt(1, 0, 5.0);
                        jacobian.setElementAt(1, 1, Math.cos(x[1]));
                    }

                    @Override
                    public int getNumberOfVariables() {
                        return 2;
                    }
                };

        var result = new MultivariateNormalDist();
        MultivariateNormalDist.propagate(evaluator, mean, cov, result);

        // check correctness
        final var evaluation = new double[2];
        var jacobian = new Matrix(2, 2);
        evaluator.evaluate(mean, evaluation, jacobian);
        assertArrayEquals(evaluation, result.getMean(), ABSOLUTE_ERROR);
        assertTrue(result.getCovariance().equals(jacobian.multiplyAndReturnNew(cov).multiplyAndReturnNew(
                jacobian.transposeAndReturnNew()), ABSOLUTE_ERROR));

        result = MultivariateNormalDist.propagate(evaluator, mean, cov);

        // check correctness
        assertArrayEquals(evaluation, result.getMean(), ABSOLUTE_ERROR);
        assertTrue(result.getCovariance().equals(jacobian.multiplyAndReturnNew(cov).multiplyAndReturnNew(
                jacobian.transposeAndReturnNew()), ABSOLUTE_ERROR));

        result = new MultivariateNormalDist();
        MultivariateNormalDist.propagate(evaluator, dist, result);

        // check correctness
        assertArrayEquals(evaluation, result.getMean(), ABSOLUTE_ERROR);
        assertTrue(result.getCovariance().equals(jacobian.multiplyAndReturnNew(cov).multiplyAndReturnNew(
                jacobian.transposeAndReturnNew()), ABSOLUTE_ERROR));

        result = MultivariateNormalDist.propagate(evaluator, dist);

        // check correctness
        assertArrayEquals(evaluation, result.getMean(), ABSOLUTE_ERROR);
        assertTrue(result.getCovariance().equals(jacobian.multiplyAndReturnNew(cov).multiplyAndReturnNew(
                jacobian.transposeAndReturnNew()), ABSOLUTE_ERROR));

        result = new MultivariateNormalDist();
        dist.propagateThisDistribution(evaluator, result);

        // check correctness
        assertArrayEquals(evaluation, result.getMean(), ABSOLUTE_ERROR);
        assertTrue(result.getCovariance().equals(jacobian.multiplyAndReturnNew(cov).multiplyAndReturnNew(
                jacobian.transposeAndReturnNew()), ABSOLUTE_ERROR));

        result = dist.propagateThisDistribution(evaluator);

        // check correctness
        assertArrayEquals(evaluation, result.getMean(), ABSOLUTE_ERROR);
        assertTrue(result.getCovariance().equals(jacobian.multiplyAndReturnNew(cov).multiplyAndReturnNew(
                jacobian.transposeAndReturnNew()), ABSOLUTE_ERROR));

        // generate a large number of Gaussian random samples and propagate
        // through function.
        final var multiRandomizer = new MultivariateGaussianRandomizer(mean, cov);
        final var x = new double[2];
        final var y = new double[2];
        jacobian = new Matrix(2, 2);

        final var resultMean = new double[2];
        final var row = new Matrix(1, 2);
        final var col = new Matrix(2, 1);
        final var sqr = new Matrix(2, 2);
        final var sqrSum = new Matrix(2, 2);
        double[] tmp;
        for (var i = 0; i < N_SAMPLES; i++) {
            multiRandomizer.next(x);
            evaluator.evaluate(x, y, jacobian);

            tmp = Arrays.copyOf(y, 2);
            ArrayUtils.multiplyByScalar(tmp, 1.0 / (double) N_SAMPLES, tmp);
            ArrayUtils.sum(resultMean, tmp, resultMean);

            col.fromArray(y);
            row.fromArray(y);
            col.multiply(row, sqr);
            sqr.multiplyByScalar(1.0 / (double) N_SAMPLES);

            sqrSum.add(sqr);
        }

        col.fromArray(resultMean);
        row.fromArray(resultMean);
        final var sqrMean = col.multiplyAndReturnNew(row);

        final var resultCov = sqrSum.subtractAndReturnNew(sqrMean);

        final var maxMean = Math.max(ArrayUtils.max(mean), Math.abs(ArrayUtils.min(mean)));
        assertArrayEquals(resultMean, result.getMean(), RELATIVE_ERROR * maxMean);

        final var maxCov = Math.max(ArrayUtils.max(result.getCovariance().getBuffer()),
                Math.abs(ArrayUtils.min(result.getCovariance().getBuffer())));
        assertTrue(resultCov.equals(result.getCovariance(), RELATIVE_ERROR * maxCov));
    }
}
