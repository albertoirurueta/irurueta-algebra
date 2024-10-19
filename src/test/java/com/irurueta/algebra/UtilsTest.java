/*
 * Copyright (C) 2012 Alberto Irurueta Carro (alberto@irurueta.com)
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
package com.irurueta.algebra;

import com.irurueta.statistics.UniformRandomizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class UtilsTest {

    private static final double MIN_RANDOM_VALUE = 0.0;
    private static final double MAX_RANDOM_VALUE = 50.0;
    private static final int MIN_ROWS = 1;
    private static final int MAX_ROWS = 50;
    private static final int MIN_COLUMNS = 1;
    private static final int MAX_COLUMNS = 50;
    private static final int MIN_LENGTH = 1;
    private static final int MAX_LENGTH = 100;

    private static final double ROUND_ERROR = 1e-3;
    private static final double BIG_ROUND_ERROR = 1.0;
    private static final double ABSOLUTE_ERROR = 1e-6;

    private static final int TIMES = 10;

    @Test
    void testTrace() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        var trace = 0.0;

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                if (i == j) {
                    trace += m.getElementAt(i, j);
                }
            }
        }

        assertEquals(trace, Utils.trace(m), ABSOLUTE_ERROR);
    }

    @Test
    void testCond() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var decomposer = new SingularValueDecomposer(m);

        decomposer.decompose();

        final var condNumber = decomposer.getConditionNumber();

        assertEquals(condNumber, Utils.cond(m), ABSOLUTE_ERROR);
    }

    @Test
    void testRank() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var decomposer = new SingularValueDecomposer(m);
        decomposer.decompose();

        final var rank = decomposer.getRank();

        assertEquals(rank, Utils.rank(m), ABSOLUTE_ERROR);
    }

    @Test
    void testDet() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        int columns;
        var t = 0;
        do {
            columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
            t++;
        } while (rows == columns && t < TIMES);

        assertNotEquals(rows, columns);

        // Test for square matrix
        var m = Matrix.createWithUniformRandomValues(rows, rows, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var decomposer = new LUDecomposer(m);
        decomposer.decompose();

        final var det = decomposer.determinant();

        assertEquals(det, Utils.det(m), ABSOLUTE_ERROR);

        // Test for non-square matrix (Force WrongSizeException)
        final var m2 = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        assertThrows(WrongSizeException.class, () -> Utils.det(m2));
    }

    @Test
    void testSolveMatrix() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException, SingularMatrixException, RankDeficientMatrixException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_COLUMNS + 5, MAX_COLUMNS + 5);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, rows - 1);
        final var colsB = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // Test for non-singular square matrix
        final var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        final var b = Matrix.createWithUniformRandomValues(rows, colsB, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var s3 = new Matrix(rows, colsB);

        final var decomposer = new LUDecomposer(m);
        decomposer.decompose();

        var s = decomposer.solve(b);
        var s2 = Utils.solve(m, b);
        Utils.solve(m, b, s3);

        assertTrue(s.equals(s2, ABSOLUTE_ERROR));
        assertEquals(s2, s3);

        // Test for singular square matrix (Force RankDeficientMatrixException)
        final var m2 = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        assertThrows(RankDeficientMatrixException.class, () -> Utils.solve(m2, b));
        assertThrows(RankDeficientMatrixException.class, () -> Utils.solve(m2, b, s3));

        // Test for non-square (rows > columns) non-rank deficient matrix
        final var m3 = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b2 = Matrix.createWithUniformRandomValues(rows, colsB, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var s4 = new Matrix(columns, colsB);

        final var decomposer2 = new EconomyQRDecomposer(m3);
        decomposer2.decompose();

        s = decomposer2.solve(b2);
        s2 = Utils.solve(m3, b2);
        Utils.solve(m3, b2, s4);

        assertTrue(s.equals(s2, ABSOLUTE_ERROR));
        assertEquals(s2, s4);

        // Test for non-square (rows < columns) matrix (Force WrongSizeException)
        final var m4 = DecomposerHelper.getSingularMatrixInstance(columns, rows);
        final var b3 = Matrix.createWithUniformRandomValues(columns, colsB, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var s5 = new Matrix(rows, colsB);
        assertThrows(WrongSizeException.class, () -> Utils.solve(m4, b3));
        assertThrows(WrongSizeException.class, () -> Utils.solve(m4, b3, s5));

        // Test for b having different number of rows than m
        final var m5 = DecomposerHelper.getSingularMatrixInstance(rows, columns);
        final var b4 = Matrix.createWithUniformRandomValues(columns, colsB, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var s6 = new Matrix(columns, colsB);
        assertThrows(WrongSizeException.class, () -> Utils.solve(m5, b4));
        assertThrows(WrongSizeException.class, () -> Utils.solve(m5, b4, s6));
    }

    @Test
    void testSolveArray() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException, SingularMatrixException, RankDeficientMatrixException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_COLUMNS + 5, MAX_COLUMNS + 5);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, rows - 1);
        final var colsB = 1;

        Matrix s;
        double[] s2;

        // Test for non-singular square matrix
        final var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        final var b = Matrix.createWithUniformRandomValues(rows, colsB, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE).toArray();
        final var s3 = new double[rows];

        final var decomposer = new LUDecomposer(m);
        decomposer.decompose();

        s = decomposer.solve(Matrix.newFromArray(b));
        s2 = Utils.solve(m, b);
        Utils.solve(m, b, s3);

        assertArrayEquals(s.toArray(), s2, ABSOLUTE_ERROR);
        assertArrayEquals(s2, s3, 0.0);

        // Test for singular square matrix (Force RankDeficientMatrixException)
        final var m2 = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        assertThrows(DecomposerException.class, () -> Utils.solve(m2, b));
        assertThrows(DecomposerException.class, () -> Utils.solve(m2, b, s3));

        // Test for non-square (rows > columns) non-rank deficient matrix
        final var m3 = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b2 = Matrix.createWithUniformRandomValues(rows, colsB, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE).toArray();
        final var s4 = new double[columns];

        final var decomposer2 = new EconomyQRDecomposer(m3);
        decomposer2.decompose();

        s = decomposer2.solve(Matrix.newFromArray(b2));
        s2 = Utils.solve(m3, b2);
        Utils.solve(m3, b2, s4);

        assertArrayEquals(s.toArray(), s2, ABSOLUTE_ERROR);
        assertArrayEquals(s2, s4, 0.0);

        // Test for non-square (rows < columns) matrix (Force WrongSizeException)
        final var m4 = DecomposerHelper.getSingularMatrixInstance(columns, rows);
        final var b3 = Matrix.createWithUniformRandomValues(columns, colsB,
                MIN_RANDOM_VALUE, MAX_RANDOM_VALUE).toArray();
        final var s5 = new double[rows];
        assertThrows(DecomposerException.class, () -> Utils.solve(m4, b3));
        assertThrows(DecomposerException.class, () -> Utils.solve(m4, b3, s5));

        // Test for b having different number of rows than m
        final var m5 = DecomposerHelper.getSingularMatrixInstance(rows, columns);
        final var b4 = Matrix.createWithUniformRandomValues(columns, colsB,
                MIN_RANDOM_VALUE, MAX_RANDOM_VALUE).toArray();
        final var s6 = new double[columns];
        assertThrows(DecomposerException.class, () -> Utils.solve(m5, b4));
        assertThrows(DecomposerException.class, () -> Utils.solve(m5, b4, s6));
    }

    @Test
    void testNormF() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var normComputer = new FrobeniusNormComputer();

        final var norm = normComputer.getNorm(m);
        assertEquals(norm, Utils.normF(m), ABSOLUTE_ERROR);
    }

    @Test
    void testNormInf() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var normComputer = new InfinityNormComputer();

        var norm = normComputer.getNorm(m);
        assertEquals(norm, Utils.normInf(m), ABSOLUTE_ERROR);

        m = Matrix.createWithUniformRandomValues(rows, 1, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        norm = normComputer.getNorm(m.toArray());
        assertEquals(norm, Utils.normInf(m.toArray()), ABSOLUTE_ERROR);
    }

    @Test
    void testNorm2() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var decomposer = new SingularValueDecomposer(m);
        decomposer.decompose();
        var norm2 = decomposer.getNorm2();
        assertEquals(norm2, Utils.norm2(m), ABSOLUTE_ERROR);

        m = Matrix.createWithUniformRandomValues(rows, 1, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        norm2 = Utils.norm2(m);
        assertEquals(norm2, Utils.norm2(m.toArray()), ABSOLUTE_ERROR);
    }

    @Test
    void testNorm1() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var normComputer = new OneNormComputer();

        var norm = normComputer.getNorm(m);
        assertEquals(norm, Utils.norm1(m), ABSOLUTE_ERROR);


        m = Matrix.createWithUniformRandomValues(rows, 1, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        norm = normComputer.getNorm(m);
        assertEquals(norm, Utils.norm1(m.toArray()), ABSOLUTE_ERROR);
    }

    @Test
    void testInverse() throws WrongSizeException, RankDeficientMatrixException, DecomposerException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(columns + 1, MAX_ROWS + 3);

        final var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        var inverse = Utils.inverse(m);
        var identity = m.multiplyAndReturnNew(inverse);
        // Check identity is correct
        assertTrue(identity.equals(Matrix.identity(rows, rows), ROUND_ERROR));

        // Test for singular square matrix (Force RankDeficientMatrixException)
        final var m2 = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        assertThrows(RankDeficientMatrixException.class, () -> Utils.inverse(m2));

        // Test for non-square (rows > columns) non-singular matrix to find
        // pseudo-inverse, hence we use BIG_RELATIVE_ERROR to test correctness
        final var m3 = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        inverse = Utils.inverse(m3);

        identity = m3.multiplyAndReturnNew(inverse);
        // Check identity is correct
        for (var j = 0; j < rows; j++) {
            for (var i = 0; i < rows; i++) {
                if (i == j) {
                    assertEquals(1.0, identity.getElementAt(i, j), BIG_ROUND_ERROR);
                } else {
                    assertEquals(0.0, identity.getElementAt(i, j), BIG_ROUND_ERROR);
                }
            }
        }

        // Test for non-square (rows < columns) matrix (Force WrongSizeException)
        final var m4 = DecomposerHelper.getSingularMatrixInstance(columns, rows);
        assertThrows(WrongSizeException.class, () -> Utils.inverse(m4));
    }

    @Test
    void testInverse2() throws WrongSizeException, RankDeficientMatrixException, DecomposerException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(columns + 1, MAX_ROWS + 3);

        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        final var inverse = new Matrix(m);
        Utils.inverse(inverse, inverse);
        var identity = m.multiplyAndReturnNew(inverse);
        // Check identity is correct
        assertTrue(identity.equals(Matrix.identity(rows, rows), ROUND_ERROR));

        // Test for singular square matrix (Force RankDeficientMatrixException)
        final var m2 = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        assertThrows(RankDeficientMatrixException.class, () -> Utils.inverse(m2, inverse));

        // Test for non-square (rows > columns) non-singular matrix to find
        // pseudo-inverse, hence we use BIG_RELATIVE_ERROR to test correctness
        final var m3 = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var inverse2 = new Matrix(rows, columns);
        Utils.inverse(m3, inverse2);

        identity = m3.multiplyAndReturnNew(inverse2);
        // Check identity is correct
        for (var j = 0; j < rows; j++) {
            for (var i = 0; i < rows; i++) {
                if (i == j) {
                    assertEquals(1.0, identity.getElementAt(i, j), BIG_ROUND_ERROR);
                } else {
                    assertEquals(0.0, identity.getElementAt(i, j), BIG_ROUND_ERROR);
                }
            }
        }

        // Test for non-square (rows < columns) matrix (Force WrongSizeException)
        final var m4 = DecomposerHelper.getSingularMatrixInstance(columns, rows);
        assertThrows(WrongSizeException.class, () -> Utils.inverse(m4, inverse));
    }

    @Test
    void testInverse3() throws WrongSizeException, DecomposerException, RankDeficientMatrixException {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = DecomposerHelper.getNonSingularMatrixInstance(length, 1);
        final var array = m.toArray();
        final var inverse1 = Utils.pseudoInverse(m);
        final var inverse2 = new Matrix(inverse1.getRows(), inverse1.getColumns());
        Utils.inverse(array, inverse2);
        final var inverse3 = Utils.inverse(array);
        final var inverse4 = Utils.pseudoInverse(array);

        assertTrue(inverse1.equals(inverse2, ABSOLUTE_ERROR));
        assertEquals(inverse2, inverse3);
        assertEquals(inverse1, inverse4);
    }

    @Test
    void testPseudoInverse() throws WrongSizeException, DecomposerException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(columns + 1, MAX_ROWS + 3);

        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        var inverse = Utils.pseudoInverse(m);
        var identity = m.multiplyAndReturnNew(inverse);
        // Check identity is correct and that pseudo-inverse is equal to the
        // inverse
        assertTrue(identity.equals(Matrix.identity(rows, rows), ROUND_ERROR));

        // Test for singular square matrix
        m = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        inverse = Utils.pseudoInverse(m);
        identity = m.multiplyAndReturnNew(inverse);
        // Check identity is correct and that pseudo-inverse is equal to the
        // inverse
        assertTrue(identity.equals(Matrix.identity(rows, rows), BIG_ROUND_ERROR));

        // Test for non-square (rows < columns) non-singular matrix
        m = DecomposerHelper.getNonSingularMatrixInstance(columns, rows);
        inverse = Utils.pseudoInverse(m);
        identity = m.multiplyAndReturnNew(inverse);
        assertTrue(identity.equals(Matrix.identity(columns, columns), BIG_ROUND_ERROR));
    }

    @Test
    void testSkew1() throws WrongSizeException {

        final var array = new double[3];
        array[0] = 1.0;
        array[1] = 4.0;
        array[2] = 2.0;

        final var m = Utils.skewMatrix(array);

        assertEquals(-array[2], m.getElementAt(0, 1), ABSOLUTE_ERROR);
        assertEquals(array[1], m.getElementAt(0, 2), ABSOLUTE_ERROR);
        assertEquals(array[2], m.getElementAt(1, 0), ABSOLUTE_ERROR);
        assertEquals(-array[0], m.getElementAt(1, 2), ABSOLUTE_ERROR);
        assertEquals(-array[1], m.getElementAt(2, 0), ABSOLUTE_ERROR);
        assertEquals(array[0], m.getElementAt(2, 1), ABSOLUTE_ERROR);

        final var m2 = new Matrix(3, 3);
        Utils.skewMatrix(array, m2);

        assertEquals(m, m2);

        final var jacobian = new Matrix(9, 3);
        final var m3 = new Matrix(3, 3);
        Utils.skewMatrix(array, m3, jacobian);

        assertEquals(m, m3);

        assertEquals(0.0, jacobian.getElementAt(0, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(1, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(2, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(3, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(4, 0), 0.0);
        assertEquals(1.0, jacobian.getElementAt(5, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(6, 0), 0.0);
        assertEquals(-1.0, jacobian.getElementAt(7, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(8, 0), 0.0);

        assertEquals(0.0, jacobian.getElementAt(0, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(1, 1), 0.0);
        assertEquals(-1.0, jacobian.getElementAt(2, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(3, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(4, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(5, 1), 0.0);
        assertEquals(1.0, jacobian.getElementAt(6, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(7, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(8, 1), 0.0);

        assertEquals(0.0, jacobian.getElementAt(0, 2), 0.0);
        assertEquals(1.0, jacobian.getElementAt(1, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(2, 2), 0.0);
        assertEquals(-1.0, jacobian.getElementAt(3, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(4, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(5, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(6, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(7, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(8, 2), 0.0);

        // Force WrongSizeException
        final var m1 = new Matrix(1, 1);
        assertThrows(WrongSizeException.class, () -> Utils.skewMatrix(array, m3, m1));
    }

    @Test
    void testSkew2() throws WrongSizeException {
        final var m = new Matrix(3, 1);
        m.setElementAt(0, 0, 1.0);
        m.setElementAt(1, 0, 4.0);
        m.setElementAt(2, 0, 3.0);

        final var mSkew = Utils.skewMatrix(m);

        assertEquals(-m.getElementAt(2, 0), mSkew.getElementAt(0, 1), ABSOLUTE_ERROR);
        assertEquals(m.getElementAt(1, 0), mSkew.getElementAt(0, 2), ABSOLUTE_ERROR);
        assertEquals(m.getElementAt(2, 0), mSkew.getElementAt(1, 0), ABSOLUTE_ERROR);
        assertEquals(-m.getElementAt(0, 0), mSkew.getElementAt(1, 2), ABSOLUTE_ERROR);
        assertEquals(-m.getElementAt(1, 0), mSkew.getElementAt(2, 0), ABSOLUTE_ERROR);
        assertEquals(m.getElementAt(0, 0), mSkew.getElementAt(2, 1), ABSOLUTE_ERROR);


        final var m2 = new Matrix(3, 3);
        Utils.skewMatrix(m, m2);

        assertEquals(mSkew, m2);

        final var jacobian = new Matrix(9, 3);
        final var m3 = new Matrix(3, 3);
        Utils.skewMatrix(m, m3, jacobian);

        assertEquals(mSkew, m3);

        assertEquals(0.0, jacobian.getElementAt(0, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(1, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(2, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(3, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(4, 0), 0.0);
        assertEquals(1.0, jacobian.getElementAt(5, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(6, 0), 0.0);
        assertEquals(-1.0, jacobian.getElementAt(7, 0), 0.0);
        assertEquals(0.0, jacobian.getElementAt(8, 0), 0.0);

        assertEquals(0.0, jacobian.getElementAt(0, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(1, 1), 0.0);
        assertEquals(-1.0, jacobian.getElementAt(2, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(3, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(4, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(5, 1), 0.0);
        assertEquals(1.0, jacobian.getElementAt(6, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(7, 1), 0.0);
        assertEquals(0.0, jacobian.getElementAt(8, 1), 0.0);

        assertEquals(0.0, jacobian.getElementAt(0, 2), 0.0);
        assertEquals(1.0, jacobian.getElementAt(1, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(2, 2), 0.0);
        assertEquals(-1.0, jacobian.getElementAt(3, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(4, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(5, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(6, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(7, 2), 0.0);
        assertEquals(0.0, jacobian.getElementAt(8, 2), 0.0);

        // Force WrongSizeException
        final var m1 = new Matrix(1, 1);
        assertThrows(WrongSizeException.class, () -> Utils.skewMatrix(m, m3, m1));
    }

    @Test
    void testCrossProduct1() throws WrongSizeException {

        final var array1 = new double[3];
        final var array2 = new double[3];

        array1[0] = 1.0;
        array1[1] = 4.0;
        array1[2] = 2.0;

        array2[0] = 4.0;
        array2[1] = 2.0;
        array2[2] = 3.0;

        final var output = Utils.crossProduct(array1, array2);
        final var output2 = new double[3];
        Utils.crossProduct(array1, array2, output2);

        assertEquals(8.0, output[0], ABSOLUTE_ERROR);
        assertEquals(5.0, output[1], ABSOLUTE_ERROR);
        assertEquals(-14.0, output[2], ABSOLUTE_ERROR);

        assertArrayEquals(output, output2, 0.0);

        final var jacobian1 = new Matrix(3, 3);
        final var jacobian2 = new Matrix(3, 3);

        Utils.crossProduct(array1, array2, output2, jacobian1, jacobian2);

        assertEquals(Utils.skewMatrix(array1).multiplyByScalarAndReturnNew(-1.0), jacobian1);
        assertEquals(Utils.skewMatrix(array2), jacobian2);

        // Force WrongSizeException
        final var m1 = new Matrix(1, 1);
        assertThrows(WrongSizeException.class, () -> Utils.crossProduct(array1, array2, m1, jacobian2));
        final var m2 = new Matrix(1, 1);
        assertThrows(WrongSizeException.class, () -> Utils.crossProduct(array1, array2, jacobian1, m2));

        assertThrows(WrongSizeException.class, () -> Utils.crossProduct(array1, array2, new double[1], jacobian1,
                jacobian2));
        final var m3 = new Matrix(1, 1);
        assertThrows(WrongSizeException.class, () -> Utils.crossProduct(array1, array2, output2, m3, jacobian2));
        final var m4 = new Matrix(1, 1);
        assertThrows(WrongSizeException.class, () -> Utils.crossProduct(array1, array2, output2, jacobian1, m4));
    }

    @Test
    void testCrossProduct2() throws WrongSizeException {

        final var array = new double[3];
        final var m = new Matrix(3, 3);

        array[0] = 1.0;
        array[1] = 4.0;
        array[2] = 2.0;

        // first row
        m.setElementAt(0, 0, 4.0);
        m.setElementAt(1, 0, 2.0);
        m.setElementAt(2, 0, 3.0);

        // second row
        m.setElementAt(0, 1, 3.0);
        m.setElementAt(1, 1, 1.0);
        m.setElementAt(2, 1, 7.0);

        // third row
        m.setElementAt(0, 2, 8.0);
        m.setElementAt(1, 2, 1.0);
        m.setElementAt(2, 2, 3.0);

        final var output = Utils.crossProduct(array, m);

        assertEquals(8.0, output.getElementAt(0, 0), ABSOLUTE_ERROR);
        assertEquals(5.0, output.getElementAt(1, 0), ABSOLUTE_ERROR);
        assertEquals(-14.0, output.getElementAt(2, 0), ABSOLUTE_ERROR);

        assertEquals(26.0, output.getElementAt(0, 1), ABSOLUTE_ERROR);
        assertEquals(-1.0, output.getElementAt(1, 1), ABSOLUTE_ERROR);
        assertEquals(-11.0, output.getElementAt(2, 1), ABSOLUTE_ERROR);

        assertEquals(10.0, output.getElementAt(0, 2), ABSOLUTE_ERROR);
        assertEquals(13.0, output.getElementAt(1, 2), ABSOLUTE_ERROR);
        assertEquals(-31.0, output.getElementAt(2, 2), ABSOLUTE_ERROR);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> Utils.crossProduct(new double[1], m));
        final var m1 = new Matrix(1, 1);
        assertThrows(WrongSizeException.class, () -> Utils.crossProduct(array, m1));
    }

    @Test
    void testCrossProduct3() throws WrongSizeException {

        final var array = new double[3];
        final var m = new Matrix(3, 3);

        array[0] = 1.0;
        array[1] = 4.0;
        array[2] = 2.0;

        // first row
        m.setElementAt(0, 0, 4.0);
        m.setElementAt(1, 0, 2.0);
        m.setElementAt(2, 0, 3.0);

        // second row
        m.setElementAt(0, 1, 3.0);
        m.setElementAt(1, 1, 1.0);
        m.setElementAt(2, 1, 7.0);

        // third row
        m.setElementAt(0, 2, 8.0);
        m.setElementAt(1, 2, 1.0);
        m.setElementAt(2, 2, 3.0);

        final var output = new Matrix(3, 1);
        Utils.crossProduct(array, m, output);

        assertEquals(8.0, output.getElementAt(0, 0), ABSOLUTE_ERROR);
        assertEquals(5.0, output.getElementAt(1, 0), ABSOLUTE_ERROR);
        assertEquals(-14.0, output.getElementAt(2, 0), ABSOLUTE_ERROR);

        assertEquals(26.0, output.getElementAt(0, 1), ABSOLUTE_ERROR);
        assertEquals(-1.0, output.getElementAt(1, 1), ABSOLUTE_ERROR);
        assertEquals(-11.0, output.getElementAt(2, 1), ABSOLUTE_ERROR);

        assertEquals(10.0, output.getElementAt(0, 2), ABSOLUTE_ERROR);
        assertEquals(13.0, output.getElementAt(1, 2), ABSOLUTE_ERROR);
        assertEquals(-31.0, output.getElementAt(2, 2), ABSOLUTE_ERROR);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> Utils.crossProduct(new double[1], m, output));
        final var m1 = new Matrix(1, 1);
        assertThrows(WrongSizeException.class, () -> Utils.crossProduct(array, m1, output));
    }

    @Test
    void testIsSymmetric() throws WrongSizeException {
        var numValid = 0;
        for (var t = 0; t < TIMES; t++) {
            final var randomizer = new UniformRandomizer();
            final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);

            final var m = DecomposerHelper.getSymmetricMatrix(rows);

            assertTrue(Utils.isSymmetric(m));
            assertTrue(Utils.isSymmetric(m, ABSOLUTE_ERROR));

            // now make matrix non-symmetric
            m.setElementAt(0, rows - 1, m.getElementAt(0, rows - 1) + 1.0);

            if (Utils.isSymmetric(m)) {
                continue;
            }
            assertFalse(Utils.isSymmetric(m));
            assertFalse(Utils.isSymmetric(m, ABSOLUTE_ERROR));

            // but if we provide a threshold large enough, matrix will still be
            // considered to be symmetric
            assertTrue(Utils.isSymmetric(m, 1.0));

            numValid++;
            break;
        }

        assertTrue(numValid > 0);
    }

    @Test
    void testIsOrthonormalAndIsOrthogonal() throws WrongSizeException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 1, MAX_ROWS);

        var m = DecomposerHelper.getOrthonormalMatrix(rows);

        assertTrue(Utils.isOrthonormal(m));
        assertTrue(Utils.isOrthonormal(m, ABSOLUTE_ERROR));
        assertTrue(Utils.isOrthogonal(m));
        assertTrue(Utils.isOrthogonal(m, ABSOLUTE_ERROR));

        // if we scale the matrix it will no longer will be orthonormal, but it
        // will continue to be orthogonal
        m.multiplyByScalar(2.0);

        assertFalse(Utils.isOrthonormal(m));
        assertFalse(Utils.isOrthonormal(m, ABSOLUTE_ERROR));
        assertTrue(Utils.isOrthogonal(m));
        assertTrue(Utils.isOrthogonal(m, ABSOLUTE_ERROR));
        // unless threshold is large enough, in which case matrix will still be
        // considered as orthonormal
        assertTrue(Utils.isOrthonormal(m, 2.0));

        // a singular matrix won't be either orthogonal or orthonormal
        m = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        assertFalse(Utils.isOrthogonal(m));
        assertFalse(Utils.isOrthogonal(m, ABSOLUTE_ERROR));
        assertFalse(Utils.isOrthonormal(m));
        assertFalse(Utils.isOrthonormal(m, ABSOLUTE_ERROR));

        // A non-square matrix won't be orthogonal or orthonormal
        m = new Matrix(rows, rows + 1);
        assertFalse(Utils.isOrthogonal(m));
        assertFalse(Utils.isOrthogonal(m, ABSOLUTE_ERROR));
        assertFalse(Utils.isOrthonormal(m));
        assertFalse(Utils.isOrthonormal(m, ABSOLUTE_ERROR));

        // Force IllegalArgumentException (by setting negative threshold)
        final var m2 = new Matrix(rows, rows + 1);
        assertThrows(IllegalArgumentException.class, () -> Utils.isOrthogonal(m2, -ABSOLUTE_ERROR));
        assertThrows(IllegalArgumentException.class, () -> Utils.isOrthonormal(m2, -ABSOLUTE_ERROR));
    }

    @Test
    void testDotProduct() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH + 1, MAX_LENGTH);

        final var input1 = new double[length];
        final var input2 = new double[length];
        randomizer.fill(input1, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        randomizer.fill(input2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        var expectedResult = 0.0;
        for (var i = 0; i < length; i++) {
            expectedResult += input1[i] * input2[i];
        }

        var result = Utils.dotProduct(input1, input2);

        // check correctness
        assertEquals(expectedResult, result, 0.0);

        // Force IllegalArgumentException
        final var wrongArray = new double[length + 1];
        assertThrows(IllegalArgumentException.class, () -> Utils.dotProduct(input1, wrongArray));
        assertThrows(IllegalArgumentException.class, () -> Utils.dotProduct(wrongArray, input2));

        // test with jacobians
        final var jacobian1 = new Matrix(1, length);
        final var jacobian2 = new Matrix(1, length);
        result = Utils.dotProduct(input1, input2, jacobian1, jacobian2);

        // check correctness
        assertEquals(expectedResult, result, 0.0);

        assertArrayEquals(input1, jacobian1.getBuffer(), 0.0);
        assertArrayEquals(input2, jacobian2.getBuffer(), 0.0);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> Utils.dotProduct(wrongArray, input2, jacobian1, jacobian2));
        assertThrows(IllegalArgumentException.class, () -> Utils.dotProduct(input1, wrongArray, jacobian1, jacobian2));
        assertThrows(IllegalArgumentException.class, () -> Utils.dotProduct(input1, input2,
                new Matrix(1, 1), jacobian2));
        assertThrows(IllegalArgumentException.class, () -> Utils.dotProduct(input1, input2, jacobian1,
                new Matrix(1, 1)));

        // test with matrices
        final var m1 = Matrix.newFromArray(input1, false);
        final var m2 = Matrix.newFromArray(input2, true);

        result = Utils.dotProduct(m1, m2);

        // check correctness
        assertEquals(expectedResult, result, 0.0);

        // Force WrongSizeException
        final var wrongMatrix = new Matrix(length + 1, 1);
        assertThrows(WrongSizeException.class, () -> Utils.dotProduct(m1, wrongMatrix));
        assertThrows(WrongSizeException.class, () -> Utils.dotProduct(wrongMatrix, m2));

        // test with jacobians
        result = Utils.dotProduct(m1, m2, jacobian1, jacobian2);

        // check correctness
        assertEquals(expectedResult, result, 0.0);

        assertArrayEquals(m1.getBuffer(), jacobian1.getBuffer(), 0.0);
        assertArrayEquals(m2.getBuffer(), jacobian2.getBuffer(), 0.0);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> Utils.dotProduct(wrongMatrix, m2, jacobian1, jacobian2));
        assertThrows(WrongSizeException.class, () -> Utils.dotProduct(m1, wrongMatrix, jacobian1, jacobian2));

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> Utils.dotProduct(m1, m2, new Matrix(1, 1),
                jacobian2));
        assertThrows(IllegalArgumentException.class, () -> Utils.dotProduct(m1, m2, jacobian1,
                new Matrix(1, 1)));
    }

    @Test
    void testSchurc() throws WrongSizeException, DecomposerException, RankDeficientMatrixException {

        final var randomizer = new UniformRandomizer();
        final var size = randomizer.nextInt(MIN_ROWS + 1, MAX_ROWS);
        final var pos = randomizer.nextInt(1, size);

        final var m = DecomposerHelper.getSymmetricPositiveDefiniteMatrixInstance(
                DecomposerHelper.getLeftLowerTriangulatorFactor(size));

        // as defined in https://en.wikipedia.org/wiki/Schur_complement
        final var a = m.getSubmatrix(0, 0, pos - 1, pos - 1);
        final var b = m.getSubmatrix(0, pos, pos - 1, size - 1);
        final var c = m.getSubmatrix(pos, 0, size - 1, pos - 1);
        final var d = m.getSubmatrix(pos, pos, size - 1, size - 1);

        assertTrue(b.equals(c.transposeAndReturnNew(), ABSOLUTE_ERROR));


        // test 1st schurc method

        // test with pos from start, and sqrt
        var result = new Matrix(size - pos, size - pos);
        var iA = new Matrix(pos, pos);
        Utils.schurc(m, pos, true, true, result, iA);

        // check correctness
        // (result is the sqr root of the Schur complement of A)
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B,
        // but result is the sqr root of that (an upper triangle matrix)
        var result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));
        var schurc = result.transposeAndReturnNew().multiplyAndReturnNew(result);
        assertTrue(schurc.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of A
        assertEquals(pos, iA.getRows());
        assertEquals(pos, iA.getColumns());
        assertTrue(a.multiplyAndReturnNew(iA).equals(Matrix.identity(pos, pos), ABSOLUTE_ERROR));

        // test with pos from start, and no sqrt
        result = new Matrix(size - pos, size - pos);
        iA = new Matrix(pos, pos);
        Utils.schurc(m, pos, true, false, result, iA);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of A
        assertEquals(pos, iA.getRows());
        assertEquals(pos, iA.getColumns());
        assertTrue(a.multiplyAndReturnNew(iA).equals(Matrix.identity(pos, pos), ABSOLUTE_ERROR));

        // test with pos from end, with sqrt
        result = new Matrix(pos, pos);
        iA = new Matrix(size - pos, size - pos);
        Utils.schurc(m, pos, false, true, result, iA);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());

        // Schur complement of D is: M/D = A - B*D^-1*C
        // but result is the sqrt root of that (an upper triangle matrix)
        result2 = a.subtractAndReturnNew(b.multiplyAndReturnNew(
                Utils.inverse(d).multiplyAndReturnNew(c)));
        schurc = result.transposeAndReturnNew().multiplyAndReturnNew(result);
        assertTrue(schurc.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of D
        assertEquals(size - pos, iA.getRows());
        assertEquals(size - pos, iA.getColumns());
        assertTrue(d.multiplyAndReturnNew(iA).equals(Matrix.identity(size - pos, size - pos),
                ABSOLUTE_ERROR));

        // test with pos from end, and no sqrt
        result = new Matrix(pos, pos);
        iA = new Matrix(size - pos, size - pos);
        Utils.schurc(m, pos, false, false, result, iA);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of D
        assertEquals(size - pos, iA.getRows());
        assertEquals(size - pos, iA.getColumns());
        assertTrue(d.multiplyAndReturnNew(iA).equals(Matrix.identity(size - pos, size - pos),
                ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var wrong = new Matrix(size, size + 1);
        final var result3 = new Matrix(size - pos, size - pos);
        final var iA2 = new Matrix(pos, pos);
        assertThrows(IllegalArgumentException.class,
                () -> Utils.schurc(wrong, pos, true, true, result3, iA2));
        assertThrows(IllegalArgumentException.class,
                () -> Utils.schurc(m, size, true, true, result3, iA2));
        assertThrows(IllegalArgumentException.class,
                () -> Utils.schurc(m, 0, true, true, result3, iA2));

        // Force RankDeficientMatrixException
        final var m2 = new Matrix(size, size);
        assertThrows(RankDeficientMatrixException.class,
                () -> Utils.schurc(m2, pos, true, true, result3, iA2));

        // test 2nd schurc method

        // test with pos from start and no sqrt
        result = new Matrix(size - pos, size - pos);
        iA = new Matrix(pos, pos);
        Utils.schurc(m, pos, true, result, iA);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B,
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of A
        assertEquals(pos, iA.getRows());
        assertEquals(pos, iA.getColumns());
        assertTrue(a.multiplyAndReturnNew(iA).equals(Matrix.identity(pos, pos), ABSOLUTE_ERROR));

        // test with pos from end and no sqrt
        result = new Matrix(pos, pos);
        iA = new Matrix(size - pos, size - pos);
        Utils.schurc(m, pos, false, result, iA);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());

        // Schur complement of D is: M/D = A - B*D^-1*C
        result2 = a.subtractAndReturnNew(b.multiplyAndReturnNew(Utils.inverse(d).multiplyAndReturnNew(c)));
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of D
        assertEquals(size - pos, iA.getRows());
        assertEquals(size - pos, iA.getColumns());
        assertTrue(d.multiplyAndReturnNew(iA).equals(Matrix.identity(size - pos, size - pos),
                ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var result4 = new Matrix(pos, pos);
        final var iA3 = new Matrix(size - pos, size - pos);
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(wrong, pos, false, result4, iA3));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, size, true, result4, iA3));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, 0, false, result4, iA3));

        // Force RankDeficientMatrixException
        assertThrows(RankDeficientMatrixException.class, () -> Utils.schurc(m2, pos, true, result4, iA3));

        // test 3rd schurc method

        // test with pos from start, and no sqrt
        result = new Matrix(size - pos, size - pos);
        iA = new Matrix(pos, pos);
        Utils.schurc(m, pos, result, iA);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B,
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));

        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of A
        assertEquals(pos, iA.getRows());
        assertEquals(pos, iA.getColumns());
        assertTrue(a.multiplyAndReturnNew(iA).equals(Matrix.identity(pos, pos), ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var result5 = new Matrix(size - pos, size - pos);
        final var iA4 = new Matrix(pos, pos);
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(wrong, pos, result5, iA4));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, size, result5, iA4));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, 0, result5, iA4));

        //Force RankDeficientMatrixException
        assertThrows(RankDeficientMatrixException.class, () -> Utils.schurc(m2, pos, result5, iA4));

        // test 4th schurc method
        // (which returns new instance)

        // test with pos from start, and sqrt
        iA = new Matrix(pos, pos);
        result = Utils.schurcAndReturnNew(m, pos, true, true, iA);

        // check correctness
        // (result is the sqrt root of the Schur complement of A)
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B,
        // but result is the sqrt root of that (an upper triangle matrix)
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));
        schurc = result.transposeAndReturnNew().multiplyAndReturnNew(result);
        assertTrue(schurc.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of A
        assertEquals(pos, iA.getRows());
        assertEquals(pos, iA.getColumns());
        assertTrue(a.multiplyAndReturnNew(iA).equals(Matrix.identity(pos, pos), ABSOLUTE_ERROR));

        // test with pos from start, and no sqrt
        iA = new Matrix(pos, pos);
        result = Utils.schurcAndReturnNew(m, pos, true, false, iA);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of A
        assertEquals(pos, iA.getRows());
        assertEquals(pos, iA.getColumns());
        assertTrue(a.multiplyAndReturnNew(iA).equals(Matrix.identity(pos, pos), ABSOLUTE_ERROR));

        // test with pos from end, with sqrt
        iA = new Matrix(size - pos, size - pos);
        result = Utils.schurcAndReturnNew(m, pos, false, true, iA);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());

        // Schur complement of D is: M/D = A - B*D^-1*C
        // but result is the sqrt root of that (an upper triangle matrix)
        result2 = a.subtractAndReturnNew(b.multiplyAndReturnNew(Utils.inverse(d).multiplyAndReturnNew(c)));
        schurc = result.transposeAndReturnNew().multiplyAndReturnNew(result);
        assertTrue(schurc.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of D
        assertEquals(size - pos, iA.getRows());
        assertEquals(size - pos, iA.getColumns());
        assertTrue(d.multiplyAndReturnNew(iA).equals(Matrix.identity(size - pos, size - pos),
                ABSOLUTE_ERROR));

        // test with pos from end, and no sqrt
        iA = new Matrix(size - pos, size - pos);
        result = Utils.schurcAndReturnNew(m, pos, false, false, iA);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of D
        assertEquals(size - pos, iA.getRows());
        assertEquals(size - pos, iA.getColumns());
        assertTrue(d.multiplyAndReturnNew(iA).equals(Matrix.identity(size - pos, size - pos),
                ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var iA5 = new Matrix(size - pos, size - pos);
        assertThrows(IllegalArgumentException.class,
                () -> Utils.schurcAndReturnNew(wrong, pos, true, true, iA5));
        assertThrows(IllegalArgumentException.class,
                () -> Utils.schurcAndReturnNew(m, size, true, true, iA5));
        assertThrows(IllegalArgumentException.class,
                () -> Utils.schurcAndReturnNew(m, 0, true, true, iA5));

        // Force RankDeficientMatrixException
        assertThrows(RankDeficientMatrixException.class,
                () -> Utils.schurcAndReturnNew(m2, pos, true, true, iA5));

        // test 5th schurc method
        // (which returns new instance)

        // test with pos from start and no sqrt
        iA = new Matrix(pos, pos);
        result = Utils.schurcAndReturnNew(m, pos, true, iA);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a)).multiplyAndReturnNew(b));
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of A
        assertEquals(pos, iA.getRows());
        assertEquals(pos, iA.getColumns());
        assertTrue(a.multiplyAndReturnNew(iA).equals(Matrix.identity(pos, pos), ABSOLUTE_ERROR));

        // test with pos from end and no sqrt
        iA = new Matrix(size - pos, size - pos);
        result = Utils.schurcAndReturnNew(m, pos, false, iA);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());

        // Schur complement of D is: M/D = A - B*D^-1*C
        result2 = a.subtractAndReturnNew(b.multiplyAndReturnNew(Utils.inverse(d).multiplyAndReturnNew(c)));
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of D
        assertEquals(size - pos, iA.getRows());
        assertEquals(size - pos, iA.getColumns());
        assertTrue(d.multiplyAndReturnNew(iA).equals(Matrix.identity(size - pos, size - pos),
                ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var iA6 = new Matrix(size - pos, size - pos);
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(wrong, pos, false, iA6));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(m, size, true, iA6));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(m, 0, false, iA6));

        // Force RankDeficientMatrixException
        assertThrows(RankDeficientMatrixException.class, () -> Utils.schurcAndReturnNew(m2, pos, true, iA6));

        // test 6th schurc method
        // (which returns new instance)

        // test with pos from start, and no sqrt
        iA = new Matrix(pos, pos);
        result = Utils.schurcAndReturnNew(m, pos, iA);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));

        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // iA is the inverse of A
        assertEquals(pos, iA.getRows());
        assertEquals(pos, iA.getColumns());
        assertTrue(a.multiplyAndReturnNew(iA).equals(Matrix.identity(pos, pos), ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var iA7 = new Matrix(pos, pos);
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(wrong, pos, iA7));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(m, size, iA7));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(m, 0, iA7));

        // Force RankDeficientMatrixException
        assertThrows(RankDeficientMatrixException.class, () -> Utils.schurcAndReturnNew(m2, pos, iA7));

        // test 7th schurc method

        // test with pos from start, and sqrt
        result = new Matrix(size - pos, size - pos);
        Utils.schurc(m, pos, true, true, result);

        // check correctness
        // (result is the sqrt root of the Schur complement of A)
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B
        // but result is the sqrt root of that (an upper triangle matrix)
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));
        schurc = result.transposeAndReturnNew().multiplyAndReturnNew(result);
        assertTrue(schurc.equals(result2, ABSOLUTE_ERROR));

        // test with pos from start, and no sqrt
        result = new Matrix(size - pos, size - pos);
        Utils.schurc(m, pos, true, false, result);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // test with pos from end, with sqrt
        result = new Matrix(pos, pos);
        Utils.schurc(m, pos, false, true, result);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());

        // Schur complement of D is: M/D = A - B*D^-1*C
        // but result is the sqrt root of that (an upper triangle matrix)
        result2 = a.subtractAndReturnNew(b.multiplyAndReturnNew(Utils.inverse(d).multiplyAndReturnNew(c)));
        schurc = result.transposeAndReturnNew().multiplyAndReturnNew(result);
        assertTrue(schurc.equals(result2, ABSOLUTE_ERROR));

        // test with pos from end, and no sqrt
        result = new Matrix(pos, pos);
        Utils.schurc(m, pos, false, false, result);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var result6 = new Matrix(pos, pos);
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(wrong, pos, true, true, result6));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, size, true, true, result6));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, 0, true, true, result6));

        // test 8th schurc method

        // test with pos from start and no sqrt
        result = new Matrix(size - pos, size - pos);
        Utils.schurc(m, pos, true, result);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // test with pos from end and no sqrt
        result = new Matrix(pos, pos);
        Utils.schurc(m, pos, false, result);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());

        // Schur complement of D is: M/D = A - B*D^-1*C
        result2 = a.subtractAndReturnNew(b.multiplyAndReturnNew(Utils.inverse(d).multiplyAndReturnNew(c)));
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var result7 = new Matrix(size - pos, size - pos);
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(wrong, pos, false, result7));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, size, true, result7));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, 0, false, result7));

        // test 9th schurc method

        // test with pos from start, and no sqrt
        result = new Matrix(size - pos, size - pos);
        Utils.schurc(m, pos, result);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));

        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var result8 = new Matrix(size - pos, size - pos);
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(wrong, pos, result8));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, size, result8));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurc(m, 0, result8));

        // test 10th schurc method
        // (which returns new instance)

        // test with pos from start, and sqrt
        result = Utils.schurcAndReturnNew(m, pos, true, true);

        // check correctness
        // (result is the sqrt root of the Schur complement of A)
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B
        // but result is the sqrt root of that (an upper triangle matrix)
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));
        schurc = result.transposeAndReturnNew().multiplyAndReturnNew(result);
        assertTrue(schurc.equals(result2, ABSOLUTE_ERROR));


        // test with pos from start, and no sqrt
        result = Utils.schurcAndReturnNew(m, pos, true, false);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));


        // test with pos from end, with sqrt
        result = Utils.schurcAndReturnNew(m, pos, false, true);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());

        // Schur complement of D is: M/D = A - B*D^-1*C
        // but result is the sqrt root of that (an upper triangle matrix)
        result2 = a.subtractAndReturnNew(b.multiplyAndReturnNew(Utils.inverse(d).multiplyAndReturnNew(c)));
        schurc = result.transposeAndReturnNew().multiplyAndReturnNew(result);
        assertTrue(schurc.equals(result2, ABSOLUTE_ERROR));


        // test with pos from end, and no sqrt
        result = Utils.schurcAndReturnNew(m, pos, false, false);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class,
                () -> Utils.schurcAndReturnNew(wrong, pos, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> Utils.schurcAndReturnNew(m, size, true, true));
        assertThrows(IllegalArgumentException.class,
                () -> Utils.schurcAndReturnNew(m, 0, true, true));

        // test 11th schurc method
        // (which returns new instance)

        // test with pos from start and no sqrt
        result = Utils.schurcAndReturnNew(m, pos, true);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));


        // test with pos from end and no sqrt
        result = Utils.schurcAndReturnNew(m, pos, false);

        // check correctness
        assertEquals(pos, result.getRows());
        assertEquals(pos, result.getColumns());

        // Schur complement of D is: M/D = A - B*D^-1*C
        result2 = a.subtractAndReturnNew(b.multiplyAndReturnNew(Utils.inverse(d).multiplyAndReturnNew(c)));
        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(wrong, pos, false));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(m, size, true));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(m, 0, false));

        // test 12th schurc method
        // (which returns new instance)

        // test with pos from start, and no sqrt
        result = Utils.schurcAndReturnNew(m, pos);

        // check correctness
        assertEquals(size - pos, result.getRows());
        assertEquals(size - pos, result.getColumns());

        // Schur complement of A is: M/A = D - C*A^-1*B
        result2 = d.subtractAndReturnNew(c.multiplyAndReturnNew(Utils.inverse(a).multiplyAndReturnNew(b)));

        assertTrue(result.equals(result2, ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(wrong, pos));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(m, size));
        assertThrows(IllegalArgumentException.class, () -> Utils.schurcAndReturnNew(m, 0));
    }
}
