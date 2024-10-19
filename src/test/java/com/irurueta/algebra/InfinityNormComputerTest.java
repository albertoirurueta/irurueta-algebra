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

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class InfinityNormComputerTest {

    private static final int MIN_LIMIT = 0;
    private static final int MAX_LIMIT = 50;
    private static final int MIN_ROWS = 1;
    private static final int MAX_ROWS = 50;
    private static final int MIN_COLUMNS = 1;
    private static final int MAX_COLUMNS = 50;
    private static final int MIN_LENGTH = 1;
    private static final int MAX_LENGTH = 100;
    private static final double MIN_RANDOM_VALUE = 0;
    private static final double MAX_RANDOM_VALUE = 100;
    private static final double ABSOLUTE_ERROR = 1e-6;

    @Test
    void testGetNormType() {
        final var normComputer = new InfinityNormComputer();
        assertEquals(NormType.INFINITY_NORM, normComputer.getNormType());
    }

    @Test
    void testGetNormMatrix() throws WrongSizeException {
        final var normComputer = new InfinityNormComputer();
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        double rowSum;
        double maxRowSum = 0.0;
        final var initValue = randomizer.nextDouble(MIN_COLUMNS, MAX_COLUMNS);
        double value;

        // For random non-initialized matrix
        var m = new Matrix(rows, columns);
        for (var i = 0; i < rows; i++) {
            rowSum = 0.0;
            for (var j = 0; j < columns; j++) {
                value = randomizer.nextInt(MIN_LIMIT, MAX_LIMIT);
                m.setElementAt(i, j, value);
                rowSum += Math.abs(value);
            }

            maxRowSum = Math.max(rowSum, maxRowSum);
        }

        assertEquals(maxRowSum, normComputer.getNorm(m), ABSOLUTE_ERROR);
        assertEquals(maxRowSum, InfinityNormComputer.norm(m), ABSOLUTE_ERROR);

        // For initialized matrix
        m = new Matrix(rows, columns);
        m.initialize(initValue);

        final var norm = initValue * columns;

        assertEquals(norm, normComputer.getNorm(m), ABSOLUTE_ERROR);
        assertEquals(norm, InfinityNormComputer.norm(m), ABSOLUTE_ERROR);

        // For identity matrix
        m = Matrix.identity(rows, columns);
        assertEquals(1.0, normComputer.getNorm(m), ABSOLUTE_ERROR);
        assertEquals(1.0, InfinityNormComputer.norm(m), ABSOLUTE_ERROR);
    }

    @Test
    void testGetNormArray() {
        final var normComputer = new InfinityNormComputer();
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);
        double norm;
        final var initValue = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);

        final var v = new double[length];

        // randomly initialize vector
        for (var i = 0; i < length; i++) {
            v[i] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        }

        norm = Math.abs(v[0]);

        assertEquals(norm, normComputer.getNorm(v), ABSOLUTE_ERROR);
        assertEquals(norm, InfinityNormComputer.norm(v), ABSOLUTE_ERROR);

        Arrays.fill(v, initValue);

        norm = initValue;

        assertEquals(norm, normComputer.getNorm(v), ABSOLUTE_ERROR);
        assertEquals(norm, InfinityNormComputer.norm(v), ABSOLUTE_ERROR);
    }

    @Test
    void testNormWithJacobian() throws AlgebraException {
        final var normComputer = new InfinityNormComputer();
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);
        double norm;

        final var v = new double[length];

        // randomly initialize vector
        for (var i = 0; i < length; i++) {
            v[i] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        }

        norm = Math.abs(v[0]);

        var jacobian = new Matrix(1, length);
        assertEquals(InfinityNormComputer.norm(v, jacobian), norm, ABSOLUTE_ERROR);
        assertEquals(jacobian, Matrix.newFromArray(v).multiplyByScalarAndReturnNew(1.0 / norm).transposeAndReturnNew());

        // Force WrongSizeException
        final var m1 = new Matrix(2, length);
        assertThrows(WrongSizeException.class, () -> InfinityNormComputer.norm(v, m1));

        jacobian = new Matrix(1, length);
        assertEquals(normComputer.getNorm(v, jacobian), norm, ABSOLUTE_ERROR);
        assertEquals(jacobian, Matrix.newFromArray(v).multiplyByScalarAndReturnNew(1.0 / norm).transposeAndReturnNew());

        // Force WrongSizeException
        final var m2 = new Matrix(2, length);
        assertThrows(WrongSizeException.class, () -> normComputer.getNorm(v, m2));
    }
}
