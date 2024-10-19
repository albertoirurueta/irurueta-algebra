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

class LUDecomposerTest {

    private static final int MIN_ROWS = 3;
    private static final int MAX_ROWS = 50;
    private static final int MIN_COLUMNS = 3;
    private static final int MAX_COLUMNS = 50;

    private static final double MIN_RANDOM_VALUE = 0.0;
    private static final double MIN_RANDOM_VALUE2 = 1.0;
    private static final double MAX_RANDOM_VALUE = 100.0;

    private static final double RELATIVE_ERROR = 1.0;
    private static final double ROUND_ERROR = 1e-3;

    private static final double EPSILON = 1e-10;

    @Test
    void testConstructor() throws WrongSizeException, LockedException {
        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        // Test 1st constructor
        var decomposer = new LUDecomposer();
        assertFalse(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(DecomposerType.LU_DECOMPOSITION, decomposer.getDecomposerType());

        decomposer.setInputMatrix(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.LU_DECOMPOSITION, decomposer.getDecomposerType());

        // Test 2nd constructor
        decomposer = new LUDecomposer(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.LU_DECOMPOSITION, decomposer.getDecomposerType());
    }

    @Test
    void testGetSetInputMatrixAndIsReady() throws WrongSizeException, LockedException, NotReadyException,
            DecomposerException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new LUDecomposer();
        assertEquals(DecomposerType.LU_DECOMPOSITION, decomposer.getDecomposerType());
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
    void testDecomposer() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        // works for any rectangular matrix size with rows >= columns
        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        var decomposer = new LUDecomposer(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        decomposer.decompose();

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertTrue(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        // Check using pivoted L
        var l = decomposer.getPivottedL();
        final var u = decomposer.getU();
        final var pivot = decomposer.getPivot();

        var m2 = l.multiplyAndReturnNew(u);

        int pivotIndex;
        assertEquals(m.getRows(), m2.getRows());
        assertEquals(m.getColumns(), m2.getColumns());
        for (var j = 0; j < m2.getColumns(); j++) {
            for (var i = 0; i < m2.getRows(); i++) {
                pivotIndex = pivot[i];
                if (!Double.isNaN(m2.getElementAt(i, j))) {
                    assertEquals(m2.getElementAt(i, j), m.getElementAt(pivotIndex, j), ROUND_ERROR);
                }
            }
        }


        // Check using L : A = L * U
        l = decomposer.getL();

        m2 = l.multiplyAndReturnNew(u);

        assertEquals(m.getRows(), m2.getRows());
        assertEquals(m.getColumns(), m2.getColumns());
        for (int j = 0; j < m2.getColumns(); j++) {
            for (int i = 0; i < m2.getRows(); i++) {
                if (!Double.isNaN(m2.getElementAt(i, j))) {
                    assertEquals(m2.getElementAt(i, j), m.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }

        // Force NotReadyException
        final var decomposer2 = new LUDecomposer();
        assertThrows(NotReadyException.class, decomposer2::decompose);
    }

    @Test
    void testIsSingular() throws WrongSizeException, LockedException, NotReadyException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var decomposer = new LUDecomposer();

        // Test for square matrix
        var m = DecomposerHelper.getSingularMatrixInstance(rows, rows);

        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::isSingular);

        decomposer.decompose();

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> decomposer.isSingular(-1.0));

        assertTrue(decomposer.isSingular());

        m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        decomposer.setInputMatrix(m);
        decomposer.decompose();

        assertFalse(decomposer.isSingular());

        // Test for non-square matrix (Force WrongSizeException
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        decomposer.setInputMatrix(m);
        decomposer.decompose();

        assertThrows(WrongSizeException.class, decomposer::isSingular);
    }

    @Test
    void testGetPivottedL() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new LUDecomposer(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getPivottedL);

        decomposer.decompose();

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertTrue(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        final var l = decomposer.getPivottedL();

        // Ensure size of l is correct
        assertEquals(l.getRows(), m.getRows());
        assertEquals(l.getColumns(), m.getColumns());

        for (var j = 0; j < l.getColumns(); j++) {
            for (var i = 0; i < l.getRows(); i++) {
                if (j > i) {
                    assertEquals(0.0, l.getElementAt(i, j), ROUND_ERROR);
                } else {
                    assertEquals(1.0, Math.abs(l.getElementAt(i, j)), RELATIVE_ERROR);
                }
            }
        }
    }

    @Test
    void testGetL() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new LUDecomposer(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getL);

        decomposer.decompose();

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertTrue(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        final var pivottedL = decomposer.getPivottedL();
        final var l = decomposer.getL();
        final var pivot = decomposer.getPivot();

        assertEquals(l.getRows(), pivottedL.getRows());
        assertEquals(l.getColumns(), pivottedL.getColumns());

        int pivotIndex;
        for (var j = 0; j < l.getColumns(); j++) {
            for (var i = 0; i < l.getRows(); i++) {
                pivotIndex = pivot[i];
                assertEquals(l.getElementAt(pivotIndex, j), pivottedL.getElementAt(i, j), 0.0);
            }
        }
    }

    @Test
    void testGetU() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new LUDecomposer(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getU);

        decomposer.decompose();

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertTrue(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        final var u = decomposer.getU();

        // Ensure size of l is correct
        assertEquals(u.getRows(), m.getColumns());
        assertEquals(u.getColumns(), m.getColumns());

        for (var j = 0; j < u.getColumns(); j++) {
            for (var i = 0; i < u.getRows(); i++) {
                if (i > j) {
                    assertEquals(0.0, u.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }
    }

    @Test
    void testGetPivot() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final int length;
        final int[] pivot;

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new LUDecomposer(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getPivot);

        decomposer.decompose();

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertTrue(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        pivot = decomposer.getPivot();

        length = pivot.length;
        assertEquals(m.getRows(), length);

        for (var i = 0; i < length; i++) {
            assertTrue(pivot[i] < length);
        }
    }

    @Test
    void testDeterminant() throws WrongSizeException, NotReadyException,
            LockedException, DecomposerException, NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        // TEST FOR ONE ELEMENT MATRIX (DETERMINANT EQUAL TO THE ELEMENT)
        // Check for matrix of size (1, 1)
        var m = new Matrix(1, 1);
        m.initialize(randomizer.nextDouble(MIN_RANDOM_VALUE + 1.0, MAX_RANDOM_VALUE + 1.0));

        final var decomposer = new LUDecomposer(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::determinant);

        decomposer.decompose();

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertTrue(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        var determinant = decomposer.determinant();

        // Check determinant is equal to element located at (0, 0)
        assertEquals(m.getElementAt(0, 0), determinant, ROUND_ERROR);

        // Square matrix
        // TEST FOR NON LD MATRIX (NON_ZERO DETERMINANT)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);

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

        determinant = decomposer.determinant();

        // Check that determinant is different of zero (we give a margin of
        // epsilon to take into account possible rounding error
        assertTrue(Math.abs(determinant) > EPSILON);

        // TEST FOR LD MATRIX (ZERO DETERMINANT)
        // Initialize matrix with 2 ld rows
        m = DecomposerHelper.getSingularMatrixInstance(rows, rows);

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

        determinant = decomposer.determinant();

        // Check that determinant is equal to zero (we give a margin of epsilon
        // to take into account possible rounding error
        assertEquals(0.0, determinant, ROUND_ERROR);

        // Test for non square matrix (Force WrongSizeException)
        m = Matrix.createWithUniformRandomValues(rows, columns,
                MIN_RANDOM_VALUE + 1.0, MAX_RANDOM_VALUE + 1.0);
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

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, decomposer::determinant);
    }

    @Test
    void testSolve() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException, SingularMatrixException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = columns + randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var columns2 = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // Try for square matrix
        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);

        final var decomposer = new LUDecomposer(m);
        decomposer.decompose();

        // Force IllegalArgumentException
        final var b = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b, -1.0));

        final var s = decomposer.solve(b);

        // check that solution after calling solve matches following equation:
        // m * s = b
        final var b2 = m.multiplyAndReturnNew(s);

        assertEquals(b.getRows(), b2.getRows());
        assertEquals(b.getColumns(), b2.getColumns());
        for (var j = 0; j < b2.getColumns(); j++) {
            for (var i = 0; i < b2.getRows(); i++) {
                assertEquals(b.getElementAt(i, j), b2.getElementAt(i, j), ROUND_ERROR);
            }
        }

        // Try for non-square matrix (Throw WrongSizeException)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b3 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);

        decomposer.setInputMatrix(m);
        decomposer.decompose();

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> decomposer.solve(b3));

        // Test for singular square matrix (Throw SingularMatrixException)
        m = DecomposerHelper.getSingularMatrixInstance(rows, rows);
        final var b4 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE2, MAX_RANDOM_VALUE);
        decomposer.setInputMatrix(m);
        decomposer.decompose();

        assertThrows(SingularMatrixException.class, () -> decomposer.solve(b4));
    }
}
