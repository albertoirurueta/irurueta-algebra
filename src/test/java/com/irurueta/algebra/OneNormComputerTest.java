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

class OneNormComputerTest {

    private static final int MIN_LIMIT = 0;
    private static final int MAX_LIMIT = 50;
    private static final int MIN_ROWS = 1;
    private static final int MAX_ROWS = 50;
    private static final int MIN_COLUMNS = 1;
    private static final int MAX_COLUMNS = 50;
    private static final int MIN_LENGTH = 1;
    private static final int MAX_LENGTH = 100;
    private static final double ABSOLUTE_ERROR = 1e-6;

    @Test
    void testGetNormType() {
        final var normComputer = new OneNormComputer();
        assertNotNull(normComputer);
        assertEquals(NormType.ONE_NORM, normComputer.getNormType());
    }

    @Test
    void testGetNormMatrix() throws WrongSizeException {
        final var normComputer = new OneNormComputer();
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        double colSum;
        var maxColSum = 0.0;
        final double norm;
        double value;

        var m = new Matrix(rows, columns);
        for (var j = 0; j < columns; j++) {
            colSum = 0.0;
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);
                m.setElementAt(i, j, value);
                colSum += Math.abs(value);
            }

            maxColSum = Math.max(colSum, maxColSum);
        }

        assertEquals(maxColSum, normComputer.getNorm(m), ABSOLUTE_ERROR);
        assertEquals(maxColSum, OneNormComputer.norm(m), ABSOLUTE_ERROR);

        // For initialized matrix
        final var initValue = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);
        m.initialize(initValue);

        norm = initValue * rows;
        assertEquals(norm, normComputer.getNorm(m), ABSOLUTE_ERROR);
        assertEquals(norm, OneNormComputer.norm(m), ABSOLUTE_ERROR);

        // For identity matrix
        m = Matrix.identity(rows, columns);
        assertEquals(1.0, normComputer.getNorm(m), ABSOLUTE_ERROR);
        assertEquals(1.0, OneNormComputer.norm(m), ABSOLUTE_ERROR);
    }

    @Test
    void testGetNormArray() {
        final var normComputer = new OneNormComputer();
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);
        var sum = 0.0;
        double norm;
        final var initValue = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);
        double value;

        final var v = new double[length];
        for (var i = 0; i < length; i++) {
            value = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);
            v[i] = value;
            sum += Math.abs(value);
        }

        norm = sum;
        assertEquals(norm, normComputer.getNorm(v), ABSOLUTE_ERROR);
        assertEquals(norm, OneNormComputer.norm(v), ABSOLUTE_ERROR);

        Arrays.fill(v, initValue);

        norm = initValue * length;

        assertEquals(norm, normComputer.getNorm(v), ABSOLUTE_ERROR);
        assertEquals(norm, OneNormComputer.norm(v), ABSOLUTE_ERROR);
    }

    @Test
    void testNormWithJacobian() throws AlgebraException {
        final var normComputer = new OneNormComputer();
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);
        var sum = 0.0;
        final double norm;
        double value;

        final var v = new double[length];
        for (var i = 0; i < length; i++) {
            value = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);
            v[i] = value;
            sum += Math.abs(value);
        }

        norm = sum;

        var jacobian = new Matrix(1, length);
        assertEquals(norm, OneNormComputer.norm(v, jacobian), ABSOLUTE_ERROR);
        assertEquals(jacobian, Matrix.newFromArray(v).
                multiplyByScalarAndReturnNew(1.0 / norm).transposeAndReturnNew());

        // Force WrongSizeException
        final var wrong = new Matrix(2, length);
        assertThrows(WrongSizeException.class, () -> OneNormComputer.norm(v, wrong));

        jacobian = new Matrix(1, length);
        assertEquals(norm, normComputer.getNorm(v, jacobian), ABSOLUTE_ERROR);
        assertEquals(jacobian, Matrix.newFromArray(v).
                multiplyByScalarAndReturnNew(1.0 / norm).
                transposeAndReturnNew());

        // Force WrongSizeException
        final var m = new Matrix(2, length);
        assertThrows(WrongSizeException.class, () -> normComputer.getNorm(v, m));

        // test zero norm
        final var v2 = new double[length];
        assertEquals(0.0, OneNormComputer.norm(v2, jacobian), 0.0);
        for (var i = 0; i < 1; i++) {
            for (var j = 0; j < length; j++) {
                assertEquals(Double.MAX_VALUE, jacobian.getElementAt(i, j), 0.0);
            }
        }
    }
}
