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

class QRDecomposerTest {

    private static final int MIN_ROWS = 1;
    private static final int MAX_ROWS = 50;
    private static final int MIN_COLUMNS = 1;
    private static final int MAX_COLUMNS = 50;
    private static final double MIN_RANDOM_VALUE = 0.0;
    private static final double MIN_RANDOM_VALUE2 = 1.0;
    private static final double MAX_RANDOM_VALUE = 100.0;

    private static final double ABSOLUTE_ERROR = 1e-6;
    private static final double RELATIVE_ERROR = 0.35;
    private static final double VALID_RATIO = 0.25;

    @Test
    void testConstructor() throws WrongSizeException, LockedException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        // Test 1st constructor
        var decomposer = new QRDecomposer();

        assertFalse(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(DecomposerType.QR_DECOMPOSITION, decomposer.getDecomposerType());

        decomposer.setInputMatrix(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.QR_DECOMPOSITION, decomposer.getDecomposerType());

        // Test 2nd constructor
        decomposer = new QRDecomposer(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.QR_DECOMPOSITION, decomposer.getDecomposerType());
    }

    @Test
    void testGetSetInputMatrixAndIsReady() throws WrongSizeException, LockedException, NotReadyException,
            DecomposerException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = randomizer.nextInt(columns + 1, MAX_ROWS + 2);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new QRDecomposer();
        assertEquals(DecomposerType.QR_DECOMPOSITION, decomposer.getDecomposerType());
        assertFalse(decomposer.isReady());

        decomposer.setInputMatrix(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        decomposer.decompose();

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertTrue(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        // when setting a new input matrix, decomposition becomes unavailable and
        // must be recomputed
        decomposer.setInputMatrix(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
    }

    @Test
    void testDecompose() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = randomizer.nextInt(columns + 1, MAX_ROWS + 2);

        Matrix m;
        final Matrix q;
        final Matrix r;
        final Matrix m2;

        m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new QRDecomposer(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        decomposer.decompose();

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertTrue(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        // Check decomposition
        q = decomposer.getQ();
        r = decomposer.getR();

        m2 = q.multiplyAndReturnNew(r);

        assertEquals(m.getRows(), m2.getRows());
        assertEquals(m.getColumns(), m2.getColumns());
        for (var j = 0; j < m2.getColumns(); j++) {
            for (var i = 0; i < m2.getRows(); i++) {
                assertEquals(m2.getElementAt(i, j), m.getElementAt(i, j), ABSOLUTE_ERROR);
            }
        }

        // Force DecomposerException
        m = Matrix.createWithUniformRandomValues(columns, rows, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        decomposer.setInputMatrix(m);
        assertThrows(DecomposerException.class, decomposer::decompose);

        // Force NotReadyException
        final var decomposer2 = new QRDecomposer();
        assertThrows(NotReadyException.class, decomposer2::decompose);
    }

    @Test
    void testIsFullRank() throws WrongSizeException, LockedException, NotReadyException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS);
        final var rows = randomizer.nextInt(columns, MAX_ROWS + 3);

        Matrix m;

        final var decomposer = new QRDecomposer();

        // Test for any rectangular or square matrix that a matrix has full rank
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::isFullRank);

        decomposer.decompose();

        // Force IllegalArgumentException with a negative round error
        assertThrows(IllegalArgumentException.class, () -> decomposer.isFullRank(-1.0));

        assertTrue(decomposer.isFullRank(ABSOLUTE_ERROR));

        // Test false case only for square matrix, for other sizes unreliable
        // results might be obtained because of rounding error
        m = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        decomposer.setInputMatrix(m);
        decomposer.decompose();

        assertFalse(decomposer.isFullRank(ABSOLUTE_ERROR));
    }

    @Test
    void testGetH() throws WrongSizeException, LockedException, NotReadyException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(columns + 1, MAX_ROWS + 4);

        final Matrix m;
        final Matrix r;

        final var decomposer = new QRDecomposer();

        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getR);

        decomposer.decompose();
        r = decomposer.getR();

        assertEquals(rows, r.getRows());
        assertEquals(columns, r.getColumns());

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                if (i > j) {
                    assertEquals(0.0, r.getElementAt(i, j), ABSOLUTE_ERROR);
                }
            }
        }
    }

    @Test
    void testGetR() throws WrongSizeException, LockedException, NotReadyException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(columns + 1, MAX_ROWS + 4);

        Matrix m;
        Matrix q;
        Matrix qTransposed;
        Matrix test;

        final var decomposer = new QRDecomposer();

        // Test for non-square matrix having rows > columns
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getQ);

        decomposer.decompose();

        q = decomposer.getQ();

        assertEquals(rows, q.getRows());
        assertEquals(rows, q.getColumns());

        // Q is an orthogonal matrix, which means that Q * Q' = I
        qTransposed = q.transposeAndReturnNew();

        test = qTransposed.multiplyAndReturnNew(q);

        assertEquals(rows, test.getRows());
        assertEquals(rows, test.getColumns());

        // Check that test is similar to identity
        assertTrue(test.equals(Matrix.identity(rows, rows), ABSOLUTE_ERROR));

        // Test for square matrix
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getQ);

        decomposer.decompose();

        q = decomposer.getQ();

        assertEquals(rows, q.getRows());
        assertEquals(rows, q.getColumns());

        // Q is an orthogonal matrix, which means that Q * Q' = I
        qTransposed = q.transposeAndReturnNew();

        test = qTransposed.multiplyAndReturnNew(q);

        assertEquals(rows, test.getRows());
        assertEquals(rows, test.getColumns());

        // Check that test is similar to identity
        assertTrue(test.equals(Matrix.identity(rows, rows), ABSOLUTE_ERROR));
    }

    @Test
    void testSolve() throws WrongSizeException, RankDeficientMatrixException, NotReadyException, LockedException,
            DecomposerException, NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 4, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, rows - 1);
        final var columns2 = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // Try for square matrix
        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        final var b = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);

        final var decomposer = new QRDecomposer(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, () -> decomposer.solve(b));
        decomposer.decompose();

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b, -1.0));

        var s = decomposer.solve(b);

        // Check that solution after calling solve matches following equation:
        // m * s = b
        var b2 = m.multiplyAndReturnNew(s);

        assertEquals(b.getRows(), b2.getRows());
        assertEquals(b.getColumns(), b2.getColumns());

        assertTrue(b.equals(b, ABSOLUTE_ERROR));

        // Try for overdetermined system (rows > columns)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b1 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);

        decomposer.setInputMatrix(m);
        decomposer.decompose();

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b1, -1.0));

        s = decomposer.solve(b);

        // check that solution after calling solve matches following equation:
        // m * s = b
        b2 = m.multiplyAndReturnNew(s);

        assertEquals(b.getRows(), b2.getRows());
        assertEquals(b.getColumns(), b2.getColumns());

        double relError;
        var valid = 0;
        final var total = b2.getColumns() * b2.getRows();
        for (var j = 0; j < b2.getColumns(); j++) {
            for (var i = 0; i < b2.getRows(); i++) {
                relError = Math.abs(RELATIVE_ERROR * b2.getElementAt(i, j));
                if (Math.abs(b2.getElementAt(i, j) - b.getElementAt(i, j)) < relError) {
                    valid++;
                }
            }
        }

        assertTrue(((double) valid / (double) total) > VALID_RATIO);

        // Try for b matrix having different number of rows than m
        // (Throws WrongSizeException)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b3 = Matrix.createWithUniformRandomValues(columns, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        assertThrows(WrongSizeException.class, () -> decomposer.solve(b3));

        // Test for Rank deficient matrix only for squared matrices
        // (for other sizes, rank deficiency might not be detected and solve
        // method would execute)
        m = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        final var b4 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        assertThrows(RankDeficientMatrixException.class, () -> decomposer.solve(b4, ABSOLUTE_ERROR));
    }

    @Test
    void testSolve2() throws WrongSizeException, RankDeficientMatrixException, NotReadyException, LockedException,
            DecomposerException, NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 4, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, rows - 1);
        final var columns2 = randomizer.nextInt(MIN_COLUMNS + 1, MAX_COLUMNS);

        // Try for square matrix
        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        final var b = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);

        final var decomposer = new QRDecomposer(m);

        final var s = new Matrix(rows, columns2);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, () -> decomposer.solve(b, s));
        decomposer.decompose();

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b, -1.0, s));

        decomposer.solve(b, s);

        // Check that solution after calling solve matches following equation:
        // m * s = b
        var b2 = m.multiplyAndReturnNew(s);

        assertEquals(b.getRows(), b2.getRows());
        assertEquals(b.getColumns(), b2.getColumns());

        assertTrue(b.equals(b, ABSOLUTE_ERROR));

        // Try for overdetermined system (rows > columns)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b3 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        final var s2 = new Matrix(columns, columns2);

        decomposer.setInputMatrix(m);
        decomposer.decompose();

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b3, -1.0, s2));

        decomposer.solve(b, s);

        // check that solution after calling solve matches following equation:
        // m * s = b
        b2 = m.multiplyAndReturnNew(s);

        assertEquals(b.getRows(), b2.getRows());
        assertEquals(b.getColumns(), b2.getColumns());

        var valid = 0;
        final var total = b2.getColumns() * b2.getRows();
        double relError;
        for (var j = 0; j < b2.getColumns(); j++) {
            for (var i = 0; i < b2.getRows(); i++) {
                relError = Math.abs(RELATIVE_ERROR * b2.getElementAt(i, j));
                if (Math.abs(b2.getElementAt(i, j) - b.getElementAt(i, j)) < relError) {
                    valid++;
                }
            }
        }

        assertTrue(((double) valid / (double) total) > VALID_RATIO);

        // Try for b matrix having different number of rows than m
        // (Throws WrongSizeException)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b4 = Matrix.createWithUniformRandomValues(columns, columns2,
                MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        final var s3 = new Matrix(columns, columns2);
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        assertThrows(WrongSizeException.class, () -> decomposer.solve(b4, s3));

        // Test for Rank deficient matrix only for squared matrices
        // (for other sizes, rank deficiency might not be detected and solve
        // method would execute)
        m = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        final var b5 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        final var s4 = new Matrix(rows, columns2);
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        assertThrows(RankDeficientMatrixException.class, () -> decomposer.solve(b5, ABSOLUTE_ERROR, s4));
    }
}
