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

class SingularValueDecomposerTest {

    private static final int MIN_ROWS = 1;
    private static final int MAX_ROWS = 50;
    private static final int MIN_COLUMNS = 1;
    private static final int MAX_COLUMNS = 50;

    private static final double MIN_RANDOM_VALUE = 0.0;
    private static final double MAX_RANDOM_VALUE = 100.0;

    private static final int MIN_ITERS = 2;
    private static final int MAX_ITERS = 50;

    private static final double RELATIVE_ERROR = 3.0;
    private static final double RELATIVE_ERROR_OVERDETERMINED = 0.35;
    private static final double ABSOLUTE_ERROR = 1e-6;
    private static final double VALID_RATIO = 0.2;
    private static final double ROUND_ERROR = 1e-3;

    private static final double EPS = 1e-12;

    private static final int TIMES = 10;

    @Test
    void testConstructor() throws WrongSizeException, LockedException, NotReadyException, DecomposerException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        var decomposer = new SingularValueDecomposer();

        assertFalse(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(DecomposerType.SINGULAR_VALUE_DECOMPOSITION,
                decomposer.getDecomposerType());
        assertEquals(SingularValueDecomposer.DEFAULT_MAX_ITERS, decomposer.getMaxIterations());

        decomposer.setInputMatrix(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.SINGULAR_VALUE_DECOMPOSITION, decomposer.getDecomposerType());

        decomposer = new SingularValueDecomposer(SingularValueDecomposer.DEFAULT_MAX_ITERS + 1);

        assertFalse(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(DecomposerType.SINGULAR_VALUE_DECOMPOSITION, decomposer.getDecomposerType());
        assertEquals(SingularValueDecomposer.DEFAULT_MAX_ITERS + 1, decomposer.getMaxIterations());

        decomposer.setInputMatrix(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(DecomposerType.SINGULAR_VALUE_DECOMPOSITION, decomposer.getDecomposerType());

        decomposer = new SingularValueDecomposer(m);
        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
        assertEquals(SingularValueDecomposer.DEFAULT_MAX_ITERS, decomposer.getMaxIterations());

        decomposer.decompose();

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertTrue(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());

        // when setting a new input matrix, decomposition becomes unavailable
        // must be recomputed
        decomposer.setInputMatrix(m);

        assertTrue(decomposer.isReady());
        assertFalse(decomposer.isLocked());
        assertFalse(decomposer.isDecompositionAvailable());
        assertEquals(m, decomposer.getInputMatrix());
    }

    @Test
    void testGetSetInputMatrix() throws WrongSizeException, LockedException, NotReadyException, DecomposerException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new SingularValueDecomposer();
        assertEquals(DecomposerType.SINGULAR_VALUE_DECOMPOSITION, decomposer.getDecomposerType());
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

        // when setting a new input matrix, decomposition becomes unavailable
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

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = randomizer.nextInt(columns, MAX_ROWS + 1);

        final var decomposer = new SingularValueDecomposer();

        // randomly initialize m
        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        // Force NotReadyException
        assertThrows(NotReadyException.class, decomposer::decompose);

        decomposer.setInputMatrix(m);
        decomposer.decompose();

        final var u = decomposer.getU();
        final var w = decomposer.getW();
        final var v = decomposer.getV();
        final var vTrans = v.transposeAndReturnNew();

        // check that w is diagonal with descending singular values
        assertEquals(columns, w.getRows());
        assertEquals(columns, w.getColumns());
        var prevSingularValue = Double.MAX_VALUE;
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < columns; i++) {
                if (i == j) {
                    assertTrue(w.getElementAt(i, j) <= prevSingularValue);
                    prevSingularValue = w.getElementAt(i, j);
                } else {
                    assertEquals(0.0, w.getElementAt(i, j), 0.0);
                }
            }
        }


        final var m2 = u.multiplyAndReturnNew(w.multiplyAndReturnNew(vTrans));

        // check that m2 is equal (except for rounding errors to m
        assertEquals(rows, m2.getRows());
        assertEquals(columns, m2.getColumns());
        assertTrue(m.equals(m2, ABSOLUTE_ERROR));
    }

    @Test
    void testGetSetMaxIterations() throws WrongSizeException, LockedException, NotReadyException {

        for (var t = 0; t < TIMES; t++) {
            final var randomizer = new UniformRandomizer();
            var maxIters = randomizer.nextInt(MIN_ITERS, MAX_ITERS);
            final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
            final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

            final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

            final var decomposer = new SingularValueDecomposer(m);

            // Check default value
            assertEquals(SingularValueDecomposer.DEFAULT_MAX_ITERS, decomposer.getMaxIterations());

            // Try before decomposing
            decomposer.setMaxIterations(maxIters);
            assertEquals(maxIters, decomposer.getMaxIterations());

            try {
                decomposer.decompose();
            } catch (DecomposerException ex) {
                continue;
            }

            // Try after decomposing
            maxIters = randomizer.nextInt(MIN_ITERS, MAX_ITERS);
            decomposer.setMaxIterations(maxIters);
            assertEquals(maxIters, decomposer.getMaxIterations());

            // Try on constructor
            final var decomposer2 = new SingularValueDecomposer(m, maxIters);
            assertEquals(maxIters, decomposer2.getMaxIterations());

            // Force IllegalArgumentException
            assertThrows(IllegalArgumentException.class, () -> decomposer2.setMaxIterations(0));

            break;
        }
    }

    @Test
    void testGetU() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();

        // Works for any matrix size
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = randomizer.nextInt(columns, MAX_ROWS + 1);

        // Randomly initialize m
        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var decomposer = new SingularValueDecomposer(m);

        decomposer.decompose();

        final var u = decomposer.getU();
        final var uTrans = u.transposeAndReturnNew();

        // Check that U is orthogonal: U' * U = I
        final var ident = uTrans.multiplyAndReturnNew(u);
        assertEquals(rows, u.getRows());
        assertEquals(columns, u.getColumns());
        for (var j = 0; j < ident.getColumns(); j++) {
            for (var i = 0; i < ident.getRows(); i++) {
                if (i == j) {
                    assertEquals(1.0, ident.getElementAt(i, j), RELATIVE_ERROR);
                } else {
                    assertEquals(0.0, ident.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }
    }

    @Test
    void testGetV() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        // Works for any matrix size
        final var row = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // Randomly initialize m
        final var m = Matrix.createWithUniformRandomValues(row, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final Matrix v;
        final Matrix vTrans;

        final var decomposer = new SingularValueDecomposer(m);

        decomposer.decompose();

        v = decomposer.getV();
        vTrans = v.transposeAndReturnNew();

        // Check that V is orthogonal: V' * V = I
        final var ident = vTrans.multiplyAndReturnNew(v);
        assertEquals(columns, v.getRows());
        assertEquals(columns, v.getColumns());
        for (var j = 0; j < ident.getColumns(); j++) {
            for (var i = 0; i < ident.getRows(); i++) {
                if (i == j) {
                    assertEquals(1.0, ident.getElementAt(i, j), RELATIVE_ERROR);
                } else {
                    assertEquals(0.0, ident.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }
    }

    @Test
    void testGetSingularValues() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        // Works for any matrix size
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // Randomly initialize m
        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final double[] singularValues;
        final var decomposer = new SingularValueDecomposer(m);

        decomposer.decompose();

        singularValues = decomposer.getSingularValues();

        // Check that singular values are ordered from largest to smallest
        assertEquals(singularValues.length, columns);
        for (var i = 1; i < columns; i++) {
            assertTrue(singularValues[i] <= singularValues[i - 1]);
            // Algorithm computes positive singular values
            assertTrue(singularValues[i] >= 0.0);
            assertTrue(singularValues[i - 1] >= 0.0);
        }
    }

    @Test
    void testGetW() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        // Works for any matrix size
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // Randomly initialize m
        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final Matrix w;
        final var w2 = new Matrix(columns, columns);
        final var decomposer = new SingularValueDecomposer(m);

        decomposer.decompose();

        w = decomposer.getW();
        decomposer.getW(w2);

        // Check that singular values are ordered from largest to smallest and
        // that W is diagonal
        assertEquals(w.getRows(), columns);
        assertEquals(w.getColumns(), columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < columns; i++) {
                if (i == j) {
                    if (i >= 1) {
                        assertTrue(w.getElementAt(i, j) <= w.getElementAt(i - 1, j - 1));
                        //Algorithm computes positive singular values
                        assertTrue(w.getElementAt(i, j) >= 0.0);
                        assertTrue(w.getElementAt(i - 1, j - 1) >= 0.0);
                    }
                } else {
                    assertEquals(0.0, w.getElementAt(i, j), ROUND_ERROR);
                }
            }
        }

        assertEquals(w, w2);

        final var m1 = new Matrix(columns + 1, columns);
        assertThrows(WrongSizeException.class, () -> decomposer.getW(m1));
        final var m2 = new Matrix(columns, columns + 1);
        assertThrows(WrongSizeException.class, () -> decomposer.getW(m2));
    }

    @Test
    void testGetNorm2() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var normComputer = new FrobeniusNormComputer();

        // works for any matrix size
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // Randomly initialize m
        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final double normFro;
        final double norm2;

        final var decomposer = new SingularValueDecomposer(m);

        final var mTrans = m.transposeAndReturnNew();
        final var m2 = mTrans.multiplyAndReturnNew(m);
        normFro = Math.sqrt(normComputer.getNorm(m2));

        decomposer.decompose();
        norm2 = decomposer.getNorm2();
        assertEquals(norm2, normFro, norm2 * RELATIVE_ERROR_OVERDETERMINED);
    }

    @Test
    void testGetConditionNumber() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        // Works for any matrix size
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var decomposer = new SingularValueDecomposer(m);
        final double[] w;
        final double condNumber;
        final double recCondNumber;

        decomposer.decompose();
        condNumber = decomposer.getConditionNumber();
        recCondNumber = decomposer.getReciprocalConditionNumber();
        w = decomposer.getSingularValues();

        if (recCondNumber > EPS) {
            assertEquals(condNumber, 1.0 / recCondNumber, condNumber * RELATIVE_ERROR);
        }

        if (w[0] >= EPS && w[columns - 1] >= EPS) {
            assertEquals(w[columns - 1] / w[0], recCondNumber, ABSOLUTE_ERROR);
        }
    }

    @Test
    void testGetRank() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = randomizer.nextInt(columns, MAX_ROWS + 1);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        var decomposer = new SingularValueDecomposer(m);

        decomposer.decompose();
        final var u = decomposer.getU();
        final var w = decomposer.getW();
        final var v = decomposer.getV();
        final var vTrans = v.transposeAndReturnNew();

        // Randomly set some singular values to zero
        int rank = columns;
        for (var i = 0; i < columns; i++) {
            if (randomizer.nextInt(0, 2) == 0) {
                // Set singular value with 50% probability
                w.setElementAt(i, i, 0.0);
                rank--;
            }
        }

        final var m2 = u.multiplyAndReturnNew(w.multiplyAndReturnNew(vTrans));
        decomposer = new SingularValueDecomposer(m2);
        decomposer.decompose();

        assertEquals(decomposer.getRank(), rank);
    }

    @Test
    void testGetNullity() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var rows = randomizer.nextInt(columns, MAX_ROWS + 1);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        var decomposer = new SingularValueDecomposer(m);

        decomposer.decompose();
        final var u = decomposer.getU();
        final var w = decomposer.getW();
        final var v = decomposer.getV();
        final var vTrans = v.transposeAndReturnNew();

        // Randomly set some singular values to zero
        int nullity = 0;
        for (var i = 0; i < columns; i++) {
            if (randomizer.nextInt(0, 2) == 0) {
                // Set singular value with 50% probability
                w.setElementAt(i, i, 0.0);
                nullity++;
            }
        }

        final var m2 = u.multiplyAndReturnNew(w.multiplyAndReturnNew(vTrans));
        decomposer = new SingularValueDecomposer(m2);
        decomposer.decompose();

        assertEquals(decomposer.getNullity(), nullity);
    }

    @Test
    void testGetRange() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 3, MAX_COLUMNS + 3);
        final var rows = randomizer.nextInt(columns, MAX_ROWS + 4);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var decomposer = new SingularValueDecomposer(m);
        final var r2 = new Matrix(1, 1);
        final Matrix rTrans;
        final Matrix ident;

        decomposer.decompose();
        final var u = decomposer.getU();
        final var w = decomposer.getW();
        final var v = decomposer.getV();
        final var vTrans = v.transposeAndReturnNew();

        // Randomly set some singular values to zero
        int rank = columns;
        for (var i = 0; i < columns; i++) {
            if (randomizer.nextInt(0, 2) == 0) {
                // Set singular value with 50% probability
                w.setElementAt(i, i, 0.0);
                rank--;
            }
        }

        final var m2 = u.multiplyAndReturnNew(w.multiplyAndReturnNew(vTrans));
        final var decomposer2 = new SingularValueDecomposer(m2);
        decomposer2.decompose();

        if (rank == 0) {
            assertThrows(NotAvailableException.class, decomposer2::getRange);
            assertThrows(NotAvailableException.class, () -> decomposer2.getRange(r2));
        } else {
            final var r = decomposer2.getRange();
            decomposer2.getRange(r2);
            rTrans = r.transposeAndReturnNew();
            ident = rTrans.multiplyAndReturnNew(r);
            assertEquals(rank, r.getColumns());
            assertEquals(rows, r.getRows());
            assertEquals(r2, r);

            for (var j = 0; j < ident.getColumns(); j++) {
                for (var i = 0; i < ident.getRows(); i++) {
                    if (i == j) {
                        assertEquals(1.0, ident.getElementAt(i, j), RELATIVE_ERROR);
                    } else {
                        assertEquals(0.0, ident.getElementAt(i, j), ROUND_ERROR);
                    }
                }
            }
        }
    }

    @Test
    void testGetNullspace() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);
        final var rows = randomizer.nextInt(columns, MAX_ROWS + 3);

        final var m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        var decomposer = new SingularValueDecomposer(m);
        final var ns2 = new Matrix(1, 1);

        decomposer.decompose();
        final var u = decomposer.getU();
        final var w = decomposer.getW();
        final var v = decomposer.getV();
        final var vTrans = v.transposeAndReturnNew();

        // Randomly set some singular values to zero
        var nullity = 0;
        for (var i = 0; i < columns; i++) {
            if (randomizer.nextInt(0, 2) == 0) {
                // Set singular value with 50% probability
                w.setElementAt(i, i, 0.0);
                nullity++;
            }
        }

        final var m2 = u.multiplyAndReturnNew(w.multiplyAndReturnNew(vTrans));

        final var decomposer2 = new SingularValueDecomposer(m2);
        decomposer2.decompose();

        if (nullity == 0) {
            assertThrows(NotAvailableException.class, decomposer2::getNullspace);
            assertThrows(NotAvailableException.class, () -> decomposer2.getNullspace(new Matrix(1, 1)));
        } else {
            final var ns = decomposer2.getNullspace();
            decomposer2.getNullspace(ns2);
            final var nsTrans = ns.transposeAndReturnNew();
            final var ident = nsTrans.multiplyAndReturnNew(ns);
            assertEquals(nullity, ns.getColumns());
            assertEquals(columns, ns.getRows());
            assertEquals(ns2, ns);

            for (var j = 0; j < ident.getColumns(); j++) {
                for (var i = 0; i < ident.getRows(); i++) {
                    if (i == j) {
                        assertEquals(1.0, ident.getElementAt(i, j), RELATIVE_ERROR);
                    } else {
                        assertEquals(0.0, ident.getElementAt(i, j), ROUND_ERROR);
                    }
                }
            }
        }
    }

    @Test
    void testSolveMatrix() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 4, MAX_ROWS + 4);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, rows - 1);
        final var columns2 = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        double relError;

        // Try for square matrix
        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        final var b = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        var s2 = new Matrix(1, 1);

        final var decomposer = new SingularValueDecomposer(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, () -> decomposer.solve(b));
        assertThrows(NotAvailableException.class, () -> decomposer.solve(b, s2));

        decomposer.decompose();
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b, -1.0));
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b, -1.0, s2));

        var s = decomposer.solve(b);
        decomposer.solve(b, s2);

        // check that solution after calling solve matches following equation:
        // m * s = b
        var b2 = m.multiplyAndReturnNew(s);

        assertEquals(b.getRows(), b2.getRows());
        assertEquals(b.getColumns(), b2.getColumns());
        assertTrue(b2.equals(b, ROUND_ERROR));
        assertEquals(s2, s);

        // Try for overdetermined system (rows > columns)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b3 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b3, -1.0));
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b3, -1.0, s2));

        s = decomposer.solve(b3);
        decomposer.solve(b3, s2);

        // check that solution after calling solve matches following equation:
        // m * s = b
        b2 = m.multiplyAndReturnNew(s);

        assertEquals(b3.getRows(), b2.getRows());
        assertEquals(b3.getColumns(), b2.getColumns());
        var valid = 0;
        final var total = b2.getColumns() * b2.getRows();
        for (var j = 0; j < b2.getColumns(); j++) {
            for (var i = 0; i < b2.getRows(); i++) {
                relError = Math.abs(RELATIVE_ERROR_OVERDETERMINED * b2.getElementAt(i, j));
                if (Math.abs(b2.getElementAt(i, j) - b.getElementAt(i, j)) < relError) {
                    valid++;
                }
            }
        }

        assertEquals(s, s2);

        assertTrue(((double) valid / (double) total) > VALID_RATIO);

        // Try for b matrix having different number of rows than m (Throws
        // WrongSizeException
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b4 = Matrix.createWithUniformRandomValues(columns, columns2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        assertThrows(WrongSizeException.class, () -> decomposer.solve(b4));
        assertThrows(WrongSizeException.class, () -> decomposer.solve(b4, s2));
    }

    @Test
    void testSolveArray() throws WrongSizeException, NotReadyException, LockedException, DecomposerException,
            NotAvailableException {

        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 4, MAX_ROWS + 4);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, rows - 1);
        final var columns2 = 1;

        double relError;

        // Try for square matrix
        var m = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        final var b = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE).toArray();
        final var s2 = new double[rows];

        final var decomposer = new SingularValueDecomposer(m);

        // Force NotAvailableException
        assertThrows(NotAvailableException.class, () -> decomposer.solve(b));
        assertThrows(NotAvailableException.class, () -> decomposer.solve(b, s2));

        decomposer.decompose();
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b, -1.0));
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b, -1.0, s2));

        var s = decomposer.solve(b);
        decomposer.solve(b, s2);

        // check that solution after calling solve matches following equation:
        // m * s = b
        var b2 = m.multiplyAndReturnNew(Matrix.newFromArray(s));

        assertEquals(b2.getRows(), b.length);
        assertEquals(1, b2.getColumns());
        assertTrue(b2.equals(Matrix.newFromArray(b), ROUND_ERROR));
        assertArrayEquals(s, s2, 0.0);

        // Try for overdetermined system (rows > columns)
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b3 = Matrix.createWithUniformRandomValues(rows, columns2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE).toArray();
        final var s3 = new double[columns];
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b3, -1.0));
        assertThrows(IllegalArgumentException.class, () -> decomposer.solve(b3, -1.0, s3));

        s = decomposer.solve(b3);
        decomposer.solve(b3, s3);

        // check that solution after calling solve matches following equation:
        // m * s = b
        b2 = m.multiplyAndReturnNew(Matrix.newFromArray(s3));

        assertEquals(b.length, b2.getRows());
        assertEquals(1, b2.getColumns());
        var valid = 0;
        final var  total = b2.getColumns() * b2.getRows();
        for (var j = 0; j < b2.getColumns(); j++) {
            for (var i = 0; i < b2.getRows(); i++) {
                relError = Math.abs(RELATIVE_ERROR_OVERDETERMINED * b2.getElementAt(i, j));
                if (Math.abs(b2.getElementAt(i, j) - b[i]) < relError) {
                    valid++;
                }
            }
        }

        assertArrayEquals(s3, s, 0.0);

        assertTrue(((double) valid / (double) total) > VALID_RATIO);

        // Try for b matrix having different number of rows than m (Throws
        // WrongSizeException
        m = DecomposerHelper.getNonSingularMatrixInstance(rows, columns);
        final var b4 = Matrix.createWithUniformRandomValues(columns, columns2,
                MIN_RANDOM_VALUE, MAX_RANDOM_VALUE).toArray();
        decomposer.setInputMatrix(m);
        decomposer.decompose();
        assertThrows(WrongSizeException.class, () -> decomposer.solve(b4));
        assertThrows(WrongSizeException.class, () -> decomposer.solve(b4, s2));
    }
}
