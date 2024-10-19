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

class RQDecomposerTest {

    private static final int MIN_ROWS = 1;
    private static final int MAX_ROWS = 50;

    private static final int MIN_COLUMNS = 1;
    private static final int MAX_COLUMNS = 50;

    private static final int MIN_RANDOM_VALUE = 0;
    private static final int MAX_RANDOM_VALUE = 100;

    private static final double RELATIVE_ERROR = 1.0;
    private static final double ROUND_ERROR = 1e-3;

    @Test
    void testConstructor() throws WrongSizeException, LockedException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        var decomposer = new RQDecomposer();

        assertFalse(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(DecomposerType.RQ_DECOMPOSITION, decomposer.getDecomposerType());

        decomposer.setInputMatrix(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.RQ_DECOMPOSITION, decomposer.getDecomposerType());

        decomposer = new RQDecomposer(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.RQ_DECOMPOSITION, decomposer.getDecomposerType());
    }

    @Test
    void testGetSetInputMatrix() throws WrongSizeException, LockedException, NotReadyException, DecomposerException {

        // RQ decomposition works for any rectangular matrix size
        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 1, MAX_COLUMNS + 1);
        final var rows = randomizer.nextInt(MIN_ROWS, columns);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new RQDecomposer();
        assertEquals(DecomposerType.RQ_DECOMPOSITION, decomposer.getDecomposerType());
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

        // When setting a new input matrix, decomposition becomes unavailable
        // and must be recomputed
        decomposer.setInputMatrix(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
    }

    @Test
    void testDecompose() throws WrongSizeException, LockedException, DecomposerException, NotReadyException,
            NotAvailableException {
        // Works for any rectangular matrix size having rows < columns (it also
        // works for square matrices)
        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(MIN_ROWS, columns - 1);

        var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new RQDecomposer();

        // Force NotReadyException
        assertThrows(NotReadyException.class, decomposer::decompose);

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

        // Check decomposition
        final var  r = decomposer.getR();
        final var q = decomposer.getQ();

        final var m2 = r.multiplyAndReturnNew(q);

        assertEquals(m2.getRows(), m.getRows());
        assertEquals(m2.getColumns(), m.getColumns());
        assertTrue(m.equals(m2, ROUND_ERROR));

        // Force DecomposerException
        m = Matrix.createWithUniformRandomValues(columns, rows, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        decomposer.setInputMatrix(m);
        assertThrows(DecomposerException.class, decomposer::decompose);
    }

    @Test
    void testGetR() throws WrongSizeException, LockedException, NotReadyException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(MIN_ROWS, columns - 1);

        final var decomposer = new RQDecomposer();

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getR);

        decomposer.decompose();
        final var r = decomposer.getR();

        assertEquals(rows, r.getRows());
        assertEquals(columns, r.getColumns());

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                if (i > j) {
                    assertEquals(0.0, r.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }
    }

    @Test
    void testGetQ() throws WrongSizeException, LockedException, NotReadyException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(MIN_ROWS, columns - 1);

        final var decomposer = new RQDecomposer();

        // Test for non-square matrix having rows <= columns
        var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        decomposer.setInputMatrix(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, decomposer::getQ);

        decomposer.decompose();
        var q = decomposer.getQ();

        assertEquals(columns, q.getRows());
        assertEquals(columns, q.getColumns());

        // Q is an orthogonal matrix, which mean that Q * Q' = I
        var qTransposed = q.transposeAndReturnNew();

        var test = qTransposed.multiplyAndReturnNew(q);

        assertEquals(columns, test.getRows());
        assertEquals(columns, test.getColumns());

        // Check that test is similar to identity
        for (var j = 0; j < rows; j++) {
            for (var i = 0; i < rows; i++) {
                if (i == j) {
                    assertEquals(1.0, Math.abs(test.getElementAt(i, j)), RELATIVE_ERROR);
                } else {
                    assertEquals(0.0, test.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }

        // Test for square matrix
        m = Matrix.createWithUniformRandomValues(rows, rows, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
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
        for (var j = 0; j < rows; j++) {
            for (var i = 0; i < rows; i++) {
                if (i == j) {
                    assertEquals(1.0, Math.abs(test.getElementAt(i, j)), RELATIVE_ERROR);
                } else {
                    assertEquals(0.0, test.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }
    }
}
