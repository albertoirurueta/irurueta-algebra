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
package com.irurueta.algebra;

import com.irurueta.statistics.UniformRandomizer;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class GaussJordanEliminationTest {

    public static final double MIN_RANDOM_VALUE = 0.0;
    public static final double MAX_RANDOM_VALUE = 50.0;

    public static final int MIN_COLUMNS = 1;
    public static final int MAX_COLUMNS = 50;

    public static final double ABSOLUTE_ERROR = 1e-6;

    @Test
    void testProcessMatrix() throws WrongSizeException, SingularMatrixException, RankDeficientMatrixException,
            DecomposerException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_COLUMNS + 5, MAX_COLUMNS + 5);
        final var colsB = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // Test for non-singular square matrix
        final var a = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        final var b = Matrix.createWithUniformRandomValues(rows, colsB, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var a2 = new Matrix(a);
        final var b2 = new Matrix(b);

        GaussJordanElimination.process(a2, b2);

        // check correctness
        final var invA = Utils.inverse(a);
        final var x = Utils.solve(a, b);

        assertTrue(a2.equals(invA, ABSOLUTE_ERROR));
        assertTrue(b2.equals(x, ABSOLUTE_ERROR));

        // Force WrongSizeException

        // non square matrix a
        final var a3 = new Matrix(rows, rows + 1);
        final var b3 = new Matrix(rows, colsB);
        assertThrows(WrongSizeException.class, () -> GaussJordanElimination.process(a3, b3));

        // different rows
        final var a4 = new Matrix(rows, rows);
        final var b4 = new Matrix(rows + 1, colsB);
        assertThrows(WrongSizeException.class, () -> GaussJordanElimination.process(a4, b4));

        // Force SingularMatrixException
        final var a5 = new Matrix(rows, rows);
        final var b5 = Matrix.createWithUniformRandomValues(rows, colsB, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        assertThrows(SingularMatrixException.class, () -> GaussJordanElimination.process(a5, b5));
    }

    @Test
    void testProcessArray() throws WrongSizeException, SingularMatrixException, RankDeficientMatrixException,
            DecomposerException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_COLUMNS + 5, MAX_COLUMNS + 5);

        // Test for non-singular square matrix
        var a = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);
        var b = new double[rows];
        randomizer.fill(b, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var a2 = new Matrix(a);
        final var b2 = Arrays.copyOf(b, rows);

        GaussJordanElimination.process(a2, b2);

        // check correctness
        final var invA = Utils.inverse(a);
        final var x = Utils.solve(a, b);

        assertTrue(a2.equals(invA, ABSOLUTE_ERROR));
        assertArrayEquals(x, b2, ABSOLUTE_ERROR);

        // Force WrongSizeException

        // non square matrix a
        final var a3 = new Matrix(rows, rows + 1);
        final var b3 = new double[rows];
        assertThrows(WrongSizeException.class, () -> GaussJordanElimination.process(a3, b3));

        // different lengths
        final var a4 = new Matrix(rows, rows);
        final var b4 = new double[rows + 1];
        assertThrows(WrongSizeException.class, () -> GaussJordanElimination.process(a4, b4));

        // Force SingularMatrixException
        final var a5 = new Matrix(rows, rows);
        final var b5 = new double[rows];
        assertThrows(SingularMatrixException.class, () -> GaussJordanElimination.process(a5, b5));
    }

    @Test
    void testInverse() throws WrongSizeException, SingularMatrixException, RankDeficientMatrixException,
            DecomposerException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_COLUMNS + 5, MAX_COLUMNS + 5);

        // Test for non-singular square matrix
        final var a = DecomposerHelper.getNonSingularMatrixInstance(rows, rows);

        final var a2 = new Matrix(a);

        GaussJordanElimination.inverse(a2);

        // check correctness
        final var invA = Utils.inverse(a);

        assertTrue(a2.equals(invA, ABSOLUTE_ERROR));

        // Force WrongSizeException

        // non square matrix a
        final var a3 = new Matrix(rows, rows + 1);
        assertThrows(WrongSizeException.class, () -> GaussJordanElimination.inverse(a3));

        // Force SingularMatrixException
        final var a4 = new Matrix(rows, rows);
        assertThrows(SingularMatrixException.class, () -> GaussJordanElimination.inverse(a4));
    }
}
