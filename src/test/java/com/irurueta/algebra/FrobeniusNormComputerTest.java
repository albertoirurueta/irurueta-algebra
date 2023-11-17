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
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class FrobeniusNormComputerTest {

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
    public void testGetNormType() {
        final FrobeniusNormComputer normComputer = new FrobeniusNormComputer();
        assertEquals(NormType.FROBENIUS_NORM, normComputer.getNormType());
    }

    @Test
    public void testGetNormMatrix() throws WrongSizeException {
        final FrobeniusNormComputer normComputer = new FrobeniusNormComputer();
        final UniformRandomizer randomizer = new UniformRandomizer(new Random());
        final int rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final int columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final int minSize = Math.min(rows, columns);
        double sum = 0.0;
        double norm;
        double initValue = randomizer.nextDouble(MIN_COLUMNS, MAX_COLUMNS);
        double value;

        // For random non-initialized matrix
        Matrix m = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                value = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);
                m.setElementAt(i, j, value);
                sum += value * value;
            }
        }

        norm = Math.sqrt(sum);

        assertEquals(norm, normComputer.getNorm(m), ABSOLUTE_ERROR);
        assertEquals(norm, FrobeniusNormComputer.norm(m), ABSOLUTE_ERROR);

        // For initialized matrix
        m = new Matrix(rows, columns);
        m.initialize(initValue);

        norm = initValue * Math.sqrt(rows * columns);

        assertEquals(norm, normComputer.getNorm(m), ABSOLUTE_ERROR);
        assertEquals(norm, FrobeniusNormComputer.norm(m), ABSOLUTE_ERROR);

        // For identity matrix
        m = Matrix.identity(rows, columns);
        assertEquals(Math.sqrt(minSize), normComputer.getNorm(m), ABSOLUTE_ERROR);
        assertEquals(Math.sqrt(minSize), FrobeniusNormComputer.norm(m), ABSOLUTE_ERROR);
    }

    @Test
    public void testGetNormArray() {
        final FrobeniusNormComputer normComputer = new FrobeniusNormComputer();
        final UniformRandomizer randomizer = new UniformRandomizer(new Random());
        final int length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);
        double sum = 0.0;
        double norm;
        final double initValue = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);
        double value;

        final double[] v = new double[length];

        for (int i = 0; i < length; i++) {
            value = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);
            v[i] = value;
            sum += value * value;
        }

        norm = Math.sqrt(sum);

        assertEquals(norm, normComputer.getNorm(v), ABSOLUTE_ERROR);
        assertEquals(norm, FrobeniusNormComputer.norm(v), ABSOLUTE_ERROR);

        Arrays.fill(v, initValue);

        norm = initValue * Math.sqrt(length);

        assertEquals(norm, normComputer.getNorm(v), ABSOLUTE_ERROR);
        assertEquals(norm, FrobeniusNormComputer.norm(v), ABSOLUTE_ERROR);
    }

    @Test
    public void testNormWithJacobian() throws AlgebraException {
        final FrobeniusNormComputer normComputer = new FrobeniusNormComputer();
        final UniformRandomizer randomizer = new UniformRandomizer(new Random());
        final int length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);
        double sum = 0.0;
        final double norm;
        double value;

        final double[] v = new double[length];

        for (int i = 0; i < length; i++) {
            value = randomizer.nextDouble(MIN_LIMIT, MAX_LIMIT);
            v[i] = value;
            sum += value * value;
        }

        norm = Math.sqrt(sum);

        Matrix jacobian = new Matrix(1, length);
        assertEquals(norm, FrobeniusNormComputer.norm(v, jacobian), ABSOLUTE_ERROR);
        assertEquals(Matrix.newFromArray(v).multiplyByScalarAndReturnNew(1.0 / norm)
                .transposeAndReturnNew(), jacobian);

        // Force WrongSizeException
        try {
            FrobeniusNormComputer.norm(v, new Matrix(2, length));
            fail("WrongSizeException expected but not thrown");
        } catch (final WrongSizeException ignore) {
        }


        jacobian = new Matrix(1, length);
        assertEquals(norm, normComputer.getNorm(v, jacobian), ABSOLUTE_ERROR);
        assertEquals(Matrix.newFromArray(v).multiplyByScalarAndReturnNew(1.0 / norm)
                .transposeAndReturnNew(), jacobian);

        // Force WrongSizeException
        try {
            normComputer.getNorm(v, new Matrix(2, length));
            fail("WrongSizeException expected but not thrown");
        } catch (final WrongSizeException ignore) {
        }
    }
}
