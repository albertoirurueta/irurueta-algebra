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

class EconomyQRDecomposerTest {

    private static final int MIN_ROWS = 3;
    private static final int MAX_ROWS = 50;
    private static final int MIN_COLUMNS = 3;
    private static final int MAX_COLUMNS = 50;
    private static final double MIN_RANDOM_VALUE = 0.0;
    private static final double MIN_RANDOM_VALUE2 = 1.0;
    private static final double MAX_RANDOM_VALUE = 100.0;
    private static final double ROUND_ERROR = 1e-3;
    private static final double ABSOLUTE_ERROR = 1e-6;
    private static final double RELATIVE_ERROR_OVERDETERMINED = 0.35;
    private static final double VALID_RATIO = 0.25;

    @Test
    void testConstructor() throws WrongSizeException, LockedException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        // Test 1st constructor
        var decomposer = new EconomyQRDecomposer();
        assertFalse(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(DecomposerType.QR_ECONOMY_DECOMPOSITION, decomposer.getDecomposerType());

        decomposer.setInputMatrix(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.QR_ECONOMY_DECOMPOSITION, decomposer.getDecomposerType());

        // Test 2nd constructor
        decomposer = new EconomyQRDecomposer(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.QR_ECONOMY_DECOMPOSITION, decomposer.getDecomposerType());
    }

    @Test
    void testGetSetInputMatrixAndIsReady() throws WrongSizeException, LockedException, NotReadyException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new EconomyQRDecomposer();

        assertEquals(DecomposerType.QR_ECONOMY_DECOMPOSITION, decomposer.getDecomposerType());
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

        // When setting a new input matrix, decomposition becomes unavailable and
        // must be recomputed
        decomposer.setInputMatrix(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
    }

    @Test
    void testDecompose() throws WrongSizeException, NotReadyException, LockedException, NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = randomizer.nextInt(columns, MAX_ROWS + 1);

        final Matrix q;
        final Matrix r;
        final Matrix m2;

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new EconomyQRDecomposer(m);

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
                assertEquals(m.getElementAt(i, j), m2.getElementAt(i, j), ROUND_ERROR);
            }
        }

        // Force NotReadyException
        final var notReadyDecomposer = new EconomyQRDecomposer();
        assertThrows(NotReadyException.class, notReadyDecomposer::decompose);
    }

    @Test
    void testIsFullRank() throws WrongSizeException, LockedException, NotReadyException, NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(columns + 1, MAX_ROWS + 3);

        final var decomposer = new EconomyQRDecomposer();

        // Test for any rectangular or square matrix that a matrix has full rank
        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::isFullRank);

        decomposer.decompose();

        // Force IllegalArgumentException with a negative round error
        assertThrows(IllegalArgumentException.class, () -> decomposer.isFullRank(-1.0));

        assertTrue(decomposer.isFullRank(ROUND_ERROR));

        // Test false case only for square matrix, for other sizes unreliable
        // results might be obtained because of rounding error
        m = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        decomposer.setInputMatrix(m);
        decomposer.decompose();

        assertFalse(decomposer.isFullRank(ROUND_ERROR));

        // Try for a matrix having less rows than columns to force
        // WrongSizeException
        m = DecomposerHelper.getNonSingularMatrixInstance(columns, rows);
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        assertThrows(WrongSizeException.class, decomposer::isFullRank);
    }

    @Test
    void testGetH() throws WrongSizeException, LockedException, NotReadyException, NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var decomposer = new EconomyQRDecomposer();

        final var m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getH);

        decomposer.decompose();
        final var h = decomposer.getH();

        assertEquals(h.getRows(), rows);
        assertEquals(h.getColumns(), columns);

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                if (i < j) {
                    assertEquals(0.0, h.getElementAt(i, j), 0.0);
                }
            }
        }
    }

    @Test
    void testGetR() throws WrongSizeException, LockedException, NotReadyException, NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var decomposer = new EconomyQRDecomposer();

        final var m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getR);

        decomposer.decompose();
        final var r = decomposer.getR();

        assertEquals(r.getRows(), columns);
        assertEquals(r.getColumns(), columns);

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < columns; i++) {
                if (i > j) {
                    assertEquals(0.0, r.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }
    }

    @Test
    void testGetQ() throws WrongSizeException, LockedException, NotReadyException, NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(columns + 1, MAX_ROWS + 3);

        final var decomposer = new EconomyQRDecomposer();

        // Test for non-square matrix having rows > columns
        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getQ);

        decomposer.decompose();
        var q = decomposer.getQ();

        assertEquals(q.getRows(), rows);
        assertEquals(q.getColumns(), columns);

        // Q is an orthogonal matrix, which means that Q * Q' = I
        var qTransposed = q.transposeAndReturnNew();

        var test = qTransposed.multiplyAndReturnNew(q);

        assertEquals(test.getRows(), columns);
        assertEquals(test.getColumns(), columns);

        // Check that test is similar to identity
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < columns; i++) {
                if (i == j) {
                    assertEquals(1.0, test.getElementAt(i, j), ROUND_ERROR);
                } else {
                    assertEquals(0.0, test.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }

        // Test for square matrix
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getQ);

        decomposer.decompose();
        q = decomposer.getQ();

        assertEquals(q.getRows(), rows);
        assertEquals(q.getColumns(), rows);

        // Q is an orthogonal matrix, which means that Q * Q' = I
        qTransposed = q.transposeAndReturnNew();

        test = qTransposed.multiplyAndReturnNew(q);

        assertEquals(test.getRows(), rows);
        assertEquals(test.getRows(), rows);

        // Check that test is similar to identity
        for (var j = 0; j < rows; j++) {
            for (var i = 0; i < rows; i++) {
                if (i == j) {
                    assertEquals(1.0, test.getElementAt(i, j), ROUND_ERROR);
                } else {
                    assertEquals(0.0, test.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }

        // Test for matrix having rows < columns (Throws WrongSizeException)
        m = DecomposerHelper.getNonSingularMatrixInstance(columns, rows);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getQ);

        decomposer.decompose();
        // Force WrongSizeException
        assertThrows(WrongSizeException.class, decomposer::getQ);
    }

    @Test
    void testSolve() throws WrongSizeException, RankDeficientMatrixException, NotReadyException, LockedException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 3, MAX_ROWS + 3);
        final var columns = randomizer.nextInt(MIN_COLUMNS, rows - 1);
        final var columns2 = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // Try for square matrix
        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        final var b = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);

        final var decomposer = new EconomyQRDecomposer(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, () -> decomposer.solve(b));

        decomposer.decompose();

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b, -1.0));

        var s = decomposer.solve(b);

        // check that solution after calling solve matches following equation:
        // m * s = b
        final var b2 = m.multiplyAndReturnNew(s);

        assertEquals(b2.getRows(), b.getRows());
        assertEquals(b2.getColumns(), b.getColumns());
        for (var j = 0; j < b2.getColumns(); j++) {
            for (var i = 0; i < b2.getRows(); i++) {
                assertEquals(b.getElementAt(i, j), b2.getElementAt(i, j), ABSOLUTE_ERROR);
            }
        }

        // Try for overdetermined system (rows > columns)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);

        decomposer.setInputMatrix(m);
        decomposer.decompose();

        // Force IllegalArgumentException
        final var b3 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b3, -1.0));

        s = decomposer.solve(b3);

        // check that solution after calling solve matches following equation:
        // m * s = b
        final var b4 = m.multiplyAndReturnNew(s);

        assertEquals(b4.getRows(), b.getRows());
        assertEquals(b4.getColumns(), b.getColumns());
        var valid = 0;
        final var total = b4.getColumns() * b4.getRows();
        for (var j = 0; j < b4.getColumns(); j++) {
            for (var i = 0; i < b4.getRows(); i++) {
                final var relError = Math.abs(RELATIVE_ERROR_OVERDETERMINED * b4.getElementAt(i, j));
                if (Math.abs(b4.getElementAt(i, j) - b.getElementAt(i, j)) < relError) {
                    valid++;
                }
            }
        }

        assertTrue((double) valid / (double) total > VALID_RATIO);

        // Try for matrix having rows < columns (Throws WrongSizeException)
        m = DecomposerHelper.getNonSingularMatrixInstance(columns, rows);
        decomposer.setInputMatrix(m);
        decomposer.decompose();

        // Force IllegalArgumentException
        final var b5 = Matrix.createWithUniformRandomValues(columns, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b5, -1.0));

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> decomposer.solve(b5));

        // Try for b matrix having different number of rows than m
        // (Throws WrongSizeException)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        final var b6 = Matrix.createWithUniformRandomValues(columns, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        assertThrows(WrongSizeException.class, () -> decomposer.solve(b6));

        // Test for rank deficient matrix only for squared matrices
        // (for other sizes, rank deficiency might not be detected and solve
        // method would execute)
        m = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        final var b7 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        assertThrows(RankDeficientMatrixException.class, () -> decomposer.solve(b7, ROUND_ERROR));
    }
}
