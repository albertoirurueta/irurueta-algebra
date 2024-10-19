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

import com.irurueta.SerializationHelper;
import com.irurueta.statistics.UniformRandomizer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.security.SecureRandom;

import static org.junit.jupiter.api.Assertions.*;

class MatrixTest {

    private static final int MIN_ROWS = 1;
    private static final int MAX_ROWS = 50;
    private static final int MIN_COLUMNS = 1;
    private static final int MAX_COLUMNS = 50;

    private static final double MIN_RANDOM_VALUE = 0.0;
    private static final double MAX_RANDOM_VALUE = 100.0;

    private static final int TIMES = 10000;

    private static final double MEAN = 5;
    private static final double STANDARD_DEVIATION = 100.0;

    private static final double ABSOLUTE_ERROR = 1e-9;
    private static final double RELATIVE_ERROR = 0.1;

    @Test
    void testConstructorGetRowsAndGetColumns() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        assertNotNull(m);
        assertEquals(rows, m.getRows());
        assertEquals(columns, m.getColumns());

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> new Matrix(0, 0));
    }

    @Test
    void testCopyConstructor() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m1 = new Matrix(rows, columns);
        assertNotNull(m1);
        assertEquals(rows, m1.getRows());
        assertEquals(columns, m1.getColumns());

        randomizer.fill(m1.getBuffer(), MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var m2 = new Matrix(m1);

        // check
        assertEquals(rows, m2.getRows());
        assertEquals(columns, m2.getColumns());
        assertArrayEquals(m1.getBuffer(), m2.getBuffer(), 0.0);
    }

    @Test
    void testGetSetElementAt() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        final var tmp = new double[rows][columns];
        double value;

        // initialize matrix and array to random values
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                tmp[i][j] = value;
                m.setElementAt(i, j, value);
            }
        }

        // check that matrix contains same values in array
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = tmp[i][j];
                assertEquals(value, m.getElementAt(i, j), 0.0);
            }
        }
    }

    @Test
    void testGetIndex() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        final var tmp = new double[rows][columns];
        double value;

        // initialize matrix and array to random values
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                tmp[i][j] = value;
                m.setElementAt(i, j, value);
            }
        }

        // check that matrix contains same values in array and that it
        // corresponds to computed index
        int index;
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                index = j * rows + i;
                assertEquals(index, m.getIndex(i, j));
                value = tmp[i][j];
                assertEquals(value, m.getElementAt(i, j), 0.0);
                assertEquals(value, m.getElementAtIndex(index), 0.0);
            }
        }
    }

    @Test
    void testGetSetElementAtIndex() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var length = rows * columns;
        int index;

        final var m1 = new Matrix(rows, columns);
        final var m2 = new Matrix(rows, columns);

        final var tmp = new double[length];
        double value;

        // initialize matrix and array to random value using column order
        for (var i = 0; i < length; i++) {
            value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
            tmp[i] = value;
            m1.setElementAtIndex(i, value, true);
            m2.setElementAtIndex(i, value);
        }

        // check that matrices have the same values contained in array using
        // column order
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                index = j * rows + i;
                value = tmp[index];
                assertEquals(value, m1.getElementAt(i, j), 0.0);
                assertEquals(value, m1.getElementAtIndex(index), 0.0);
                assertEquals(value, m1.getElementAtIndex(index, true), 0.0);

                assertEquals(value, m2.getElementAt(i, j), 0.0);
                assertEquals(value, m2.getElementAtIndex(index), 0.0);
                assertEquals(value, m2.getElementAtIndex(index, true), 0.0);
            }
        }

        // initialize matrix m1 and array to random values using row order
        for (var i = 0; i < length; i++) {
            value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
            tmp[i] = value;
            m1.setElementAtIndex(i, value, false);
        }

        // checks that matrix contains same values in array using row order
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                index = i * columns + j;
                value = tmp[index];
                assertEquals(value, m1.getElementAt(i, j), 0.0);
                assertEquals(value, m1.getElementAtIndex(index, false), 0.0);
            }
        }
    }

    @Test
    void testClone() throws WrongSizeException, CloneNotSupportedException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // instantiate matrix and fill with random values
        final var m1 = new Matrix(rows, columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m1.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        // clone matrix
        final var m2 = m1.clone();

        // check correctness
        assertEquals(rows, m2.getRows());
        assertEquals(columns, m2.getColumns());

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(m1.getElementAt(i, j), m2.getElementAt(i, j), 0.0);
            }
        }
    }

    @Test
    void testCopyTo() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // instantiate matrix and fill with random values
        final var m = new Matrix(rows, columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        // instantiate destination matrix
        final var destination = new Matrix(1, 1);
        assertEquals(1, destination.getRows());
        assertEquals(1, destination.getColumns());

        // copy to destination
        m.copyTo(destination);

        // check correctness
        assertEquals(rows, destination.getRows());
        assertEquals(columns, destination.getColumns());

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(destination.getElementAt(i, j), m.getElementAt(i, j), 0.0);
            }
        }

        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> m.copyTo(null));
    }

    @Test
    void testCopyFrom() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var source = new Matrix(rows, columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                source.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        // instantiate destination matrix
        final var destination = new Matrix(1, 1);
        assertEquals(1, destination.getRows());
        assertEquals(1, destination.getColumns());

        // copy from source
        destination.copyFrom(source);

        // check correctness
        assertEquals(destination.getRows(), rows);
        assertEquals(destination.getColumns(), columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(destination.getElementAt(i, j), source.getElementAt(i, j), 0.0);
            }
        }

        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> destination.copyFrom(null));
    }

    @Test
    void testAddAndReturnNew() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m1 = new Matrix(rows, columns);
        final var m2 = new Matrix(rows, columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m1.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
                m2.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        var m3 = m1.addAndReturnNew(m2);

        // check correctness
        assertEquals(rows, m3.getRows());
        assertEquals(columns, m3.getColumns());
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals( m1.getElementAt(i, j) + m2.getElementAt(i, j), m3.getElementAt(i, j),
                        ABSOLUTE_ERROR);
            }
        }

        // Force WrongSizeException
        final var wrong = new Matrix(rows + 1, columns + 1);
        assertThrows(WrongSizeException.class, () -> m1.addAndReturnNew(wrong));

        // Force NullPointerException
        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> m1.addAndReturnNew(null));
    }

    @Test
    void testAdd() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m1 = new Matrix(rows, columns);
        final var m2 = new Matrix(rows, columns);
        final var tmp = new double[rows][columns];
        double value;
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                m1.setElementAt(i, j, value);
                tmp[i][j] = value;
                m2.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        final var m3 = new Matrix(rows, columns);
        m1.add(m2, m3);
        m1.add(m2);

        // check correctness
        assertEquals(rows, m1.getRows());
        assertEquals(columns, m1.getColumns());
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals( tmp[i][j] + m2.getElementAt(i, j), m1.getElementAt(i, j), ABSOLUTE_ERROR);
            }
        }

        assertEquals(m1, m3);

        // Force WrongSizeException
        final var wrong = new Matrix(rows + 1, columns + 1);
        assertThrows(WrongSizeException.class, () -> m1.add(wrong));

        // Force NullPointerException
        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> m1.add(null));
    }

    @Test
    void testSubtractAndReturnNew() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m1 = new Matrix(rows, columns);
        final var m2 = new Matrix(rows, columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m1.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
                m2.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        var m3 = m1.subtractAndReturnNew(m2);

        // check correctness
        assertEquals(rows, m3.getRows());
        assertEquals(columns, m3.getColumns());
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals( m1.getElementAt(i, j) - m2.getElementAt(i, j), m3.getElementAt(i, j),
                        ABSOLUTE_ERROR);
            }
        }

        // Force WrongSizeException
        var wrong = new Matrix(rows + 1, columns + 1);
        assertThrows(WrongSizeException.class, () -> m1.subtractAndReturnNew(wrong));

        // Force NullPointerException
        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> m1.subtractAndReturnNew(null));
    }

    @Test
    void testSubtract() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m1 = new Matrix(rows, columns);
        final var m2 = new Matrix(rows, columns);
        final var tmp = new double[rows][columns];
        double value;
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                m1.setElementAt(i, j, value);
                tmp[i][j] = value;
                m2.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        final var m3 = new Matrix(rows, columns);
        m1.subtract(m2, m3);
        m1.subtract(m2);

        // check correctness
        assertEquals(rows, m1.getRows());
        assertEquals(columns, m1.getColumns());
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals( tmp[i][j] - m2.getElementAt(i, j), m1.getElementAt(i, j), ABSOLUTE_ERROR);
            }
        }
        assertEquals(m1, m3);

        // Force WrongSizeExceptionException
        final var wrong = new Matrix(rows + 1, columns + 1);
        assertThrows(WrongSizeException.class, () -> m1.add(wrong));

        // Force NullPointerException
        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> m1.add(null));
    }

    @Test
    void testMultiplyAndReturnNew() throws WrongSizeException {
        final var rows1 = 4;
        final var columns1 = 3;
        final var rows2 = 3;
        final var columns2 = 2;
        final var m1 = new Matrix(rows1, columns1);
        final var m2 = new Matrix(rows2, columns2);

        Matrix result;

        // fill m1 and m2 with predefined values
        m1.setElementAt(0, 0, 1.0);
        m1.setElementAt(0, 1, 2.0);
        m1.setElementAt(0, 2, 3.0);
        m1.setElementAt(1, 0, 4.0);
        m1.setElementAt(1, 1, 5.0);
        m1.setElementAt(1, 2, 6.0);
        m1.setElementAt(2, 0, 6.0);
        m1.setElementAt(2, 1, 5.0);
        m1.setElementAt(2, 2, 4.0);
        m1.setElementAt(3, 0, 3.0);
        m1.setElementAt(3, 1, 2.0);
        m1.setElementAt(3, 2, 1.0);

        m2.setElementAt(0, 0, 1.0);
        m2.setElementAt(0, 1, 2.0);
        m2.setElementAt(1, 0, 3.0);
        m2.setElementAt(1, 1, 4.0);
        m2.setElementAt(2, 0, 5.0);
        m2.setElementAt(2, 1, 6.0);

        // make matrix product
        result = m1.multiplyAndReturnNew(m2);

        // we know result for provided set of matrices m1 and m2, check it is
        // correct
        assertEquals(result.getRows(), rows1);
        assertEquals(result.getColumns(), columns2);

        assertEquals(22.0, result.getElementAt(0, 0), ABSOLUTE_ERROR);
        assertEquals(28.0, result.getElementAt(0, 1), ABSOLUTE_ERROR);
        assertEquals(49.0, result.getElementAt(1, 0), ABSOLUTE_ERROR);
        assertEquals(64.0, result.getElementAt(1, 1), ABSOLUTE_ERROR);
        assertEquals(41.0, result.getElementAt(2, 0), ABSOLUTE_ERROR);
        assertEquals(56.0, result.getElementAt(2, 1), ABSOLUTE_ERROR);
        assertEquals(14.0, result.getElementAt(3, 0), ABSOLUTE_ERROR);
        assertEquals(20.0, result.getElementAt(3, 1), ABSOLUTE_ERROR);


        // Force IllegalArgumentException
        final var m3 = new Matrix(columns1, rows1);
        final var m4 = new Matrix(columns2, rows2);

        assertThrows(WrongSizeException.class, () -> m3.multiplyAndReturnNew(m4));

        // Force NullPointerException
        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> m3.multiplyAndReturnNew(null));
    }

    @Test
    void testMultiply() throws WrongSizeException {
        final var rows1 = 4;
        final var columns1 = 3;
        final var rows2 = 3;
        final var columns2 = 2;
        final var m1 = new Matrix(rows1, columns1);
        final var m2 = new Matrix(rows2, columns2);

        // fill m1 and m2 with predefined values
        m1.setElementAt(0, 0, 1.0);
        m1.setElementAt(0, 1, 2.0);
        m1.setElementAt(0, 2, 3.0);
        m1.setElementAt(1, 0, 4.0);
        m1.setElementAt(1, 1, 5.0);
        m1.setElementAt(1, 2, 6.0);
        m1.setElementAt(2, 0, 6.0);
        m1.setElementAt(2, 1, 5.0);
        m1.setElementAt(2, 2, 4.0);
        m1.setElementAt(3, 0, 3.0);
        m1.setElementAt(3, 1, 2.0);
        m1.setElementAt(3, 2, 1.0);

        m2.setElementAt(0, 0, 1.0);
        m2.setElementAt(0, 1, 2.0);
        m2.setElementAt(1, 0, 3.0);
        m2.setElementAt(1, 1, 4.0);
        m2.setElementAt(2, 0, 5.0);
        m2.setElementAt(2, 1, 6.0);

        // make matrix product
        m1.multiply(m2);

        // we know result for provided set of matrices m1 and m2, check it is
        // correct
        assertEquals(rows1, m1.getRows());
        assertEquals(columns2, m1.getColumns());

        assertEquals(22.0, m1.getElementAt(0, 0), ABSOLUTE_ERROR);
        assertEquals(28.0, m1.getElementAt(0, 1), ABSOLUTE_ERROR);
        assertEquals(49.0, m1.getElementAt(1, 0), ABSOLUTE_ERROR);
        assertEquals(64.0, m1.getElementAt(1, 1), ABSOLUTE_ERROR);
        assertEquals(41.0, m1.getElementAt(2, 0), ABSOLUTE_ERROR);
        assertEquals(56.0, m1.getElementAt(2, 1), ABSOLUTE_ERROR);
        assertEquals(14.0, m1.getElementAt(3, 0), ABSOLUTE_ERROR);
        assertEquals(20.0, m1.getElementAt(3, 1), ABSOLUTE_ERROR);

        // Force IllegalArgumentException
        final var m3 = new Matrix(columns1, rows1);
        final var m4 = new Matrix(columns2, rows2);

        assertThrows(WrongSizeException.class, () -> m3.multiply(m4));

        // Force NullPointerException
        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> m3.multiply(null));
    }

    @Test
    void testMultiplyKroneckerAndReturnNew() throws WrongSizeException {
        final var m1 = new Matrix(2, 2);
        final var m2 = new Matrix(2, 2);

        m1.setSubmatrix(0, 0, 1, 1, new double[]{1, 3, 2, 1});
        m2.setSubmatrix(0, 0, 1, 1, new double[]{0, 2, 3, 1});

        final var m3 = m1.multiplyKroneckerAndReturnNew(m2);

        // check correctness
        assertEquals(4, m3.getRows());
        assertEquals(4, m3.getColumns());

        final var m3b = new Matrix(4, 4);
        //noinspection all
        m3b.setSubmatrix(0, 0, 3, 3, new double[]{
                1 * 0, 1 * 2, 3 * 0, 3 * 2,
                1 * 3, 1 * 1, 3 * 3, 3 * 1,
                2 * 0, 2 * 2, 1 * 0, 1 * 2,
                2 * 3, 2 * 1, 1 * 3, 1 * 1
        });

        assertEquals(m3b, m3);
    }

    @Test
    void testMultiplyKronecker() throws WrongSizeException {
        final var m1 = new Matrix(2, 2);
        final var m2 = new Matrix(2, 2);

        m1.setSubmatrix(0, 0, 1, 1, new double[]{1, 3, 2, 1});
        m2.setSubmatrix(0, 0, 1, 1, new double[]{0, 2, 3, 1});

        final var m3 = new Matrix(2, 2);
        m1.multiplyKronecker(m2, m3);
        m1.multiplyKronecker(m2);

        // check correctness
        assertEquals(4, m3.getRows());
        assertEquals(4, m3.getColumns());

        assertEquals(4, m1.getRows());
        assertEquals(4, m1.getColumns());

        final var m3b = new Matrix(4, 4);
        //noinspection all
        m3b.setSubmatrix(0, 0, 3, 3, new double[]{
                1 * 0, 1 * 2, 3 * 0, 3 * 2,
                1 * 3, 1 * 1, 3 * 3, 3 * 1,
                2 * 0, 2 * 2, 1 * 0, 1 * 2,
                2 * 3, 2 * 1, 1 * 3, 1 * 1
        });

        assertEquals(m3b, m1);
        assertEquals(m3b, m3);
    }

    @Test
    void testMultiplyByScalarAndReturnNew() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        // fill matrix
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        final var scalar = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var result = m.multiplyByScalarAndReturnNew(scalar);

        // check correctness
        assertEquals(result.getRows(), rows);
        assertEquals(result.getColumns(), columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(m.getElementAt(i, j) * scalar, result.getElementAt(i, j), ABSOLUTE_ERROR);
            }
        }
    }

    @Test
    void testMultiplyByScalar() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        final var tmp = new double[rows][columns];
        double value;
        // fill matrix
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                m.setElementAt(i, j, value);
                tmp[i][j] = value;
            }
        }

        final var scalar = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        m.multiplyByScalar(scalar);

        // check correctness
        assertEquals(m.getRows(), rows);
        assertEquals(m.getColumns(), columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(tmp[i][j] * scalar, m.getElementAt(i, j), ABSOLUTE_ERROR);
            }
        }
    }

    @Test
    void testEqualsAndHashCode() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        final var equal = new Matrix(rows, columns);
        final var different1 = new Matrix(rows + 1, columns + 1);
        final var different2 = new Matrix(rows, columns);
        final var different3 = new Object();

        double value;
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                m.setElementAt(i, j, value);
                equal.setElementAt(i, j, value);
                different1.setElementAt(i, j, value);
                different2.setElementAt(i, j, value + 1.0);
            }
        }

        // check correctness
        //noinspection EqualsWithItself
        assertTrue(m.equals(m));
        assertTrue(m.equals(equal));
        assertFalse(m.equals(different1));
        assertFalse(m.equals(different2));
        assertNotEquals(different3, m);
        assertFalse(m.equals(null));

        assertEquals(equal.hashCode(), m.hashCode());

        // check with threshold
        assertTrue(m.equals(m, ABSOLUTE_ERROR));
        assertTrue(m.equals(equal, ABSOLUTE_ERROR));
        assertFalse(m.equals(different1, ABSOLUTE_ERROR));
        assertFalse(m.equals(different2, ABSOLUTE_ERROR));
        assertFalse(m.equals(null, ABSOLUTE_ERROR));
    }

    @Test
    void testElementByElementProductAndReturnNew() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m1 = new Matrix(rows, columns);
        final var m2 = new Matrix(rows, columns);

        // fill matrices
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m1.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
                m2.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        var m3 = m1.elementByElementProductAndReturnNew(m2);

        // check correctness
        assertEquals(rows, m3.getRows());
        assertEquals(columns, m3.getColumns());
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(m1.getElementAt(i, j) * m2.getElementAt(i, j), m3.getElementAt(i, j),
                        ABSOLUTE_ERROR);
            }
        }

        // Force WrongSizeException
        final var wrong = new Matrix(rows + 1, columns + 1);
        assertThrows(WrongSizeException.class, () -> m1.elementByElementProductAndReturnNew(wrong));

        // Force NullPointerException
        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> m1.elementByElementProductAndReturnNew(null));
    }

    @Test
    void testElementByElementProduct() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m1 = new Matrix(rows, columns);
        final var m2 = new Matrix(rows, columns);
        final var tmp = new double[rows][columns];
        double value;

        // fill matrices
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                m1.setElementAt(i, j, value);
                tmp[i][j] = value;
                m2.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        final var m3 = new Matrix(rows, columns);
        m1.elementByElementProduct(m2, m3);
        m1.elementByElementProduct(m2);

        // check correctness
        assertEquals(rows, m1.getRows());
        assertEquals(columns, m1.getColumns());
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(tmp[i][j] * m2.getElementAt(i, j), m1.getElementAt(i, j), ABSOLUTE_ERROR);
            }
        }
        assertEquals(m3, m1);

        // Force WrongSizeException
        final var wrong = new Matrix(rows + 1, columns + 1);
        assertThrows(WrongSizeException.class, () -> m1.elementByElementProduct(wrong));

        // Force NullPointerException
        //noinspection DataFlowIssue
        assertThrows(NullPointerException.class, () -> m1.elementByElementProduct(null));
    }

    @Test
    void testTransposeAndReturnNEw() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m1 = new Matrix(rows, columns);
        // fill matrix
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m1.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        final var m2 = m1.transposeAndReturnNew();

        // check correctness
        assertEquals(columns, m2.getRows());
        assertEquals(rows, m2.getColumns());

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(m2.getElementAt(j, i), m1.getElementAt(i, j), 0.0);
            }
        }
    }

    @Test
    void testTranspose() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        final var tmp = new double[rows][columns];
        double value;

        // fill matrix
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                m.setElementAt(i, j, value);
                tmp[i][j] = value;
            }
        }

        final var m2 = new Matrix(rows, columns);
        m.transpose(m2);
        m.transpose();

        // check correctness
        assertEquals(columns, m.getRows());
        assertEquals(rows, m.getColumns());

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(m.getElementAt(j, i), tmp[i][j], 0.0);
            }
        }

        assertEquals(m, m2);
    }

    @Test
    void testInitialize() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        // fill with random values
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        // pick an init value
        final var value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        m.initialize(value);

        // check correctness
        assertEquals(rows, m.getRows());
        assertEquals(columns, m.getColumns());

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                assertEquals(value, m.getElementAt(i, j), 0.0);
            }
        }
    }

    @Test
    void testResetAndResize() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows1 = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var rows2 = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns1 = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var columns2 = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows1, columns1);
        assertEquals(rows1, m.getRows());
        assertEquals(columns1, m.getColumns());

        // reset to new size
        m.resize(rows2, columns2);

        // check correctness
        assertEquals(rows2, m.getRows());
        assertEquals(columns2, m.getColumns());

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> m.resize(0, columns2));
        assertThrows(WrongSizeException.class, () -> m.resize(rows2, 0));
        assertThrows(WrongSizeException.class, () -> m.resize(0, 0));

        // reset to new size and value
        final var initValue = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        // reset to new size and value
        m.reset(rows1, columns1, initValue);

        // check correctness
        assertEquals(rows1, m.getRows());
        assertEquals(columns1, m.getColumns());

        for (var j = 0; j < columns1; j++) {
            for (var i = 0; i < rows1; i++) {
                assertEquals(initValue, m.getElementAt(i, j), 0.0);
            }
        }

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> m.reset(0, columns1, initValue));
        assertThrows(WrongSizeException.class, () -> m.reset(rows1, 0, initValue));
        assertThrows(WrongSizeException.class, () -> m.reset(0, 0, initValue));
    }

    @Test
    void testToArray() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        final var array = new double[rows * columns];
        double value;
        var counter = 0;

        // fill matrix
        if (Matrix.DEFAULT_USE_COLUMN_ORDER) {
            // use column order
            for (var j = 0; j < columns; j++) {
                for (var i = 0; i < rows; i++) {
                    value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                    m.setElementAt(i, j, value);
                    array[counter] = value;
                    counter++;
                }
            }
        }

        final var array2 = m.toArray();
        final var array3 = new double[array.length];
        m.toArray(array3);

        // check correctness
        for (var i = 0; i < rows * columns; i++) {
            assertEquals(array2[i], array[i], 0.0);
            assertEquals(array3[i], array[i], 0.0);
        }
    }

    @Test
    void testToArrayWithColumnOrder() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        final var array = new double[rows * columns];
        double value;
        var counter = 0;

        // fill with column order
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                m.setElementAt(i, j, value);
                array[counter] = value;
                counter++;
            }
        }

        final var array2 = m.toArray(true);
        final var array3 = new double[array.length];
        m.toArray(array3, true);

        // check correctness
        for (var i = 0; i < rows * columns; i++) {
            assertEquals(array2[i], array[i], 0.0);
            assertEquals(array3[i], array[i], 0.0);
        }
    }

    @Test
    void testToArrayWithRowOrder() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = new Matrix(rows, columns);
        final var array = new double[rows * columns];
        double value;
        var counter = 0;

        // fill with row order
        for (var i = 0; i < rows; i++) {
            for (var j = 0; j < columns; j++) {
                value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
                m.setElementAt(i, j, value);
                array[counter] = value;
                counter++;
            }
        }

        final var array2 = m.toArray(false);
        final var array3 = new double[array.length];
        m.toArray(array3, false);

        // check correctness
        for (var i = 0; i < rows * columns; i++) {
            assertEquals(array2[i], array[i], 0.0);
            assertEquals(array3[i], array[i], 0.0);
        }
    }

    @Test
    void testGetSubmatrix() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var m = new Matrix(rows, columns);

        // fill matrix with random values
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        final var submatrix = m.getSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn);

        // check correctness
        assertEquals(bottomRightRow - topLeftRow + 1, submatrix.getRows());
        assertEquals(bottomRightColumn - topLeftColumn + 1, submatrix.getColumns());

        for (var j = 0; j < submatrix.getColumns(); j++) {
            for (var i = 0; i < submatrix.getRows(); i++) {
                assertEquals(m.getElementAt(i + topLeftRow, j + topLeftColumn),
                        submatrix.getElementAt(i, j), 0.0);
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrix(topLeftRow + 1, topLeftColumn,
                topLeftRow, topLeftColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrix(topLeftRow, topLeftColumn + 1,
                topLeftRow, topLeftColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrix(topLeftRow + 1,
                topLeftColumn + 1, topLeftRow, topLeftColumn));
    }

    @Test
    void testGetSubmatrixAsArray() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var m = new Matrix(rows, columns);

        // fill matrix with random values
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        var array = m.getSubmatrixAsArray(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn);
        assertEquals((bottomRightRow - topLeftRow + 1) * (bottomRightColumn - topLeftColumn + 1),
                array.length);
        var counter = 0;

        if (Matrix.DEFAULT_USE_COLUMN_ORDER) {
            // column order
            for (var j = 0; j < (bottomRightColumn - topLeftColumn + 1); j++) {
                for (var i = 0; i < (bottomRightRow - topLeftRow + 1); i++) {
                    assertEquals(m.getElementAt(i + topLeftRow, j + topLeftColumn), array[counter],
                            0.0);
                    counter++;
                }
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow, columns, bottomRightRow,
                bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow, topLeftColumn, rows,
                bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow, topLeftColumn,
                bottomRightRow, columns));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow + 1,
                topLeftColumn, topLeftRow, topLeftColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow,
                topLeftColumn + 1, topLeftRow, topLeftColumn));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow + 1,
                topLeftColumn + 1, topLeftRow, topLeftColumn));
    }

    @Test
    void testGetSubmatrixAsArrayWithColumnOrder() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var m = new Matrix(rows, columns);

        // fill matrix with random values
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        var array = m.getSubmatrixAsArray(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn,
                true);
        assertEquals((bottomRightRow - topLeftRow + 1) * (bottomRightColumn - topLeftColumn + 1),
                array.length);
        var counter = 0;

        // column order
        for (var j = 0; j < (bottomRightColumn - topLeftColumn + 1); j++) {
            for (var i = 0; i < (bottomRightRow - topLeftRow + 1); i++) {
                assertEquals(m.getElementAt(i + topLeftRow, j + topLeftColumn), array[counter], 0.0);
                counter++;
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(rows, topLeftColumn,
                bottomRightRow, bottomRightColumn, true));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow, columns,
                bottomRightRow, bottomRightColumn, true));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, true));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow, topLeftColumn,
                bottomRightRow, columns, true));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow + 1,
                topLeftColumn, topLeftRow, topLeftColumn, true));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow,
                topLeftColumn + 1, topLeftRow, topLeftColumn, true));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow + 1,
                topLeftColumn + 1, topLeftRow, topLeftColumn, true));
    }

    @Test
    void testGetSubmatrixAsArrayWithRowOrder() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var m = new Matrix(rows, columns);

        // fill matrix with random values
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        var array = m.getSubmatrixAsArray(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn,
                false);
        assertEquals( (bottomRightRow - topLeftRow + 1) * (bottomRightColumn - topLeftColumn + 1),
                array.length);
        var counter = 0;

        // row order
        for (var i = 0; i < (bottomRightRow - topLeftRow + 1); i++) {
            for (var j = 0; j < (bottomRightColumn - topLeftColumn + 1); j++) {
                assertEquals(m.getElementAt(i + topLeftRow, j + topLeftColumn), array[counter], 0.0);
                counter++;
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(rows, topLeftColumn,
                bottomRightRow, bottomRightColumn, false));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow, columns,
                bottomRightRow, bottomRightColumn, false));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, false));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow, topLeftColumn,
                bottomRightRow, columns, false));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow + 1,
                topLeftColumn, topLeftRow, topLeftColumn, false));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow,
                topLeftColumn + 1, topLeftRow, topLeftColumn, false));
        assertThrows(IllegalArgumentException.class, () -> m.getSubmatrixAsArray(topLeftRow + 1,
                topLeftColumn + 1, topLeftRow, topLeftColumn, false));
    }

    @Test
    void testSetSubmatrix() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var m = new Matrix(rows, columns);

        final var submatrixRows = bottomRightRow - topLeftRow + 1;
        final var submatrixColumns = bottomRightColumn - topLeftColumn + 1;
        final var submatrix = new Matrix(submatrixRows, submatrixColumns);

        // fill sub-matrix with random values
        for (var j = 0; j < submatrixColumns; j++) {
            for (var i = 0; i < submatrixRows; i++) {
                submatrix.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        // set sub-matrix
        m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, submatrix);

        // check correctness
        for (var j = 0; j < submatrixColumns; j++) {
            for (var i = 0; i < submatrixRows; i++) {
                assertEquals(submatrix.getElementAt(i, j),
                        m.getElementAt(i + topLeftRow, j + topLeftColumn), 0.0);
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(-topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, -topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn, submatrix));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, -bottomRightRow,
                bottomRightColumn, submatrix));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, submatrix));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                -bottomRightColumn, submatrix));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns, submatrix));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, submatrix));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, submatrix));

        final var wrong = new Matrix(submatrixRows + 1, submatrixColumns);
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, wrong));

        final var wrong2 = new Matrix(submatrixRows, submatrixColumns + 1);
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, wrong2));

        final var wrong3 = new Matrix(submatrixRows + 1, submatrixColumns + 1);
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, wrong3));
    }

    @Test
    void testSetSubmatrix2() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var m = new Matrix(rows, columns);

        final var submatrix = new Matrix(rows, columns);

        // fill sub-matrix with random values
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                submatrix.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        // set sub-matrix
        m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, submatrix, topLeftRow,
                topLeftColumn, bottomRightRow, bottomRightColumn);

        // check correctness
        for (var j = topLeftColumn; j <= bottomRightColumn; j++) {
            for (var i = topLeftRow; i < bottomRightRow; i++) {
                assertEquals(submatrix.getElementAt(i, j), m.getElementAt(i, j), 0.0);
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(-topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, -topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, -bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                -bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns, submatrix, topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, submatrix, topLeftRow, topLeftColumn,
                bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, -topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, rows, topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, -topLeftColumn, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, columns, bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, -bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, rows, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow, -bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow, columns));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, submatrix, topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, submatrix, topLeftRow, topLeftColumn,
                bottomRightRow, bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, m, topLeftRow, topLeftColumn, bottomRightRow + 1,
                bottomRightColumn));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, m, topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn + 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, m, topLeftRow, topLeftColumn, bottomRightRow + 1,
                bottomRightColumn + 1));
    }

    @Test
    void testSetSubmatrixWithValue() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var value = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var m = new Matrix(rows, columns);

        // fil matrix with random values
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m.setElementAt(i, j, value + 1.0);
            }
        }

        // set sub-matrix
        m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, value);

        // check correctness
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                if (i >= topLeftRow && i <= bottomRightRow && j >= topLeftColumn && j <= bottomRightColumn) {
                    assertEquals(value, m.getElementAt(i, j), 0.0);
                } else {
                    assertEquals(value + 1.0, m.getElementAt(i, j), 0.0);
                }
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(-topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, value));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn, value));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, -topLeftColumn, bottomRightRow,
                bottomRightColumn, value));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn, value));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, -bottomRightRow,
                bottomRightColumn, value));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, value));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                -bottomRightColumn, value));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns, value));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, value));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, value));
    }

    @Test
    void testSetSubmatrixWithArray() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var m = new Matrix(rows, columns);

        final var submatrixRows = bottomRightRow - topLeftRow + 1;
        final var submatrixColumns = bottomRightColumn - topLeftColumn + 1;
        final var length = submatrixRows * submatrixColumns;
        final var array = new double[length];

        // fill array with random values
        for (var i = 0; i < length; i++) {
            array[i] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        }

        // set sub-matrix
        m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, array);
        var counter = 0;

        // check correctness
        if (Matrix.DEFAULT_USE_COLUMN_ORDER) {
            // column order
            for (var j = 0; j < submatrixColumns; j++) {
                for (var i = 0; i < submatrixRows; i++) {
                    assertEquals(array[counter], m.getElementAt(i + topLeftRow, j + topLeftColumn),
                            0.0);
                    counter++;
                }
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(-topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn, array));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, -topLeftColumn, bottomRightRow,
                bottomRightColumn, array));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn, array));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, -bottomRightRow,
                bottomRightColumn, array));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, array));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                -bottomRightColumn, array));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns, array));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, array));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, array));

        final var wrong = new double[length + 1];
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, wrong));
    }

    @Test
    void testSetSubmatrixWithArrayColumnOrder() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var m = new Matrix(rows, columns);

        final var submatrixRows = bottomRightRow - topLeftRow + 1;
        final var submatrixColumns = bottomRightColumn - topLeftColumn + 1;
        final var length = submatrixRows * submatrixColumns;
        final var array = new double[length];

        // fill array with random values
        for (var i = 0; i < length; i++) {
            array[i] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        }

        // set sub-matrix
        m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, array, true);
        var counter = 0;

        // check correctness with column order
        for (var j = 0; j < submatrixColumns; j++) {
            for (var i = 0; i < submatrixRows; i++) {
                assertEquals(array[counter], m.getElementAt(i + topLeftRow, j + topLeftColumn), 0.0);
                counter++;
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(-topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, -topLeftColumn, bottomRightRow,
                bottomRightColumn, array, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn, array, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, -bottomRightRow,
                bottomRightColumn, array, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, array, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                -bottomRightColumn, array, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns, array, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, array, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, array, true));

        final var wrong = new double[length + 1];
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, wrong, true));
    }

    @Test
    void testSetSubmatrixWithArrayRowOrder() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var m = new Matrix(rows, columns);

        final var submatrixRows = bottomRightRow - topLeftRow + 1;
        final var submatrixColumns = bottomRightColumn - topLeftColumn + 1;
        final var length = submatrixRows * submatrixColumns;
        final var array = new double[length];

        // fill array with random values
        for (var i = 0; i < length; i++) {
            array[i] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        }

        // set sub-matrix
        m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, array, false);
        var counter = 0;

        // check correctness with row order
        for (var i = 0; i < submatrixRows; i++) {
            for (var j = 0; j < submatrixColumns; j++) {
                assertEquals(array[counter], m.getElementAt(i + topLeftRow, j + topLeftColumn), 0.0);
                counter++;
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(-topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, -topLeftColumn, bottomRightRow,
                bottomRightColumn, array, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn, array, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, -bottomRightRow,
                bottomRightColumn, array, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, array, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                -bottomRightColumn, array, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns, array, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, array, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, array, false));

        final var wrong = new double[length + 1];
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, wrong, false));
    }

    @Test
    void testSetSubmatrixWithArray2() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var offset = randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m = new Matrix(rows, columns);

        final var submatrixRows = bottomRightRow - topLeftRow + 1;
        final var submatrixColumns = bottomRightColumn - topLeftColumn + 1;
        final var length = submatrixRows * submatrixColumns;
        final var array = new double[length + offset];

        // fill array with random values
        for (int i = 0; i < length; i++) {
            array[i + offset] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        }

        // set sub-matrix
        m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1);
        var counter = offset;

        // check correctness
        if (Matrix.DEFAULT_USE_COLUMN_ORDER) {
            // column order
            for (var j = 0; j < submatrixColumns; j++) {
                for (var i = 0; i < submatrixRows; i++) {
                    assertEquals(array[counter], m.getElementAt(i + topLeftRow, j + topLeftColumn),
                            0.0);
                    counter++;
                }
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(-topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, -topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, -bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, array, offset, offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                -bottomRightColumn, array, offset, offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns, array, offset, offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, array, offset, offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, array, offset,
                offset + length - 1));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length));
    }

    @Test
    void testSetSubmatrixWithArrayColumnOrder2() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var offset = randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m = new Matrix(rows, columns);

        final var submatrixRows = bottomRightRow - topLeftRow + 1;
        final var submatrixColumns = bottomRightColumn - topLeftColumn + 1;
        final var length = submatrixRows * submatrixColumns;
        final var array = new double[length + offset];

        // fill array with random values
        for (var i = 0; i < length; i++) {
            array[i + offset] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        }

        // set sub-matrix
        m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, true);
        var counter = offset;

        // check correctness with column order
        for (var j = 0; j < submatrixColumns; j++) {
            for (var i = 0; i < submatrixRows; i++) {
                assertEquals(array[counter], m.getElementAt(i + topLeftRow, j + topLeftColumn), 0.0);
                counter++;
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(-topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, -topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, -bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, array, offset, offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                -bottomRightColumn, array, offset, offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns, array, offset, offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, array, offset, offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, array, offset,
                offset + length - 1, true));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length, true));
    }

    @Test
    void testSetSubmatrixWithArrayRowOrder2() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS + 2, MAX_ROWS + 2);
        final var columns = randomizer.nextInt(MIN_COLUMNS + 2, MAX_COLUMNS + 2);

        final var topLeftColumn = randomizer.nextInt(MIN_COLUMNS, columns - 1);
        final var topLeftRow = randomizer.nextInt(MIN_ROWS, rows - 1);

        final var bottomRightColumn = randomizer.nextInt(topLeftColumn, columns - 1);
        final var bottomRightRow = randomizer.nextInt(topLeftRow, rows - 1);

        final var offset = randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var m = new Matrix(rows, columns);

        final var submatrixRows = bottomRightRow - topLeftRow + 1;
        final var submatrixColumns = bottomRightColumn - topLeftColumn + 1;
        final var length = submatrixRows * submatrixColumns;
        final var array = new double[length + offset];

        // fill array with random values
        for (var i = 0; i < length; i++) {
            array[i + offset] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        }

        // set sub-matrix
        m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, array,
                offset, offset + length - 1, false);
        var counter = offset;

        // check correctness with row order
        for (var i = 0; i < submatrixRows; i++) {
            for (var j = 0; j < submatrixColumns; j++) {
                assertEquals(array[counter], m.getElementAt(i + topLeftRow, j + topLeftColumn), 0.0);
                counter++;
            }
        }

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(-topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(rows, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, -topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, columns, bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, -bottomRightRow,
                bottomRightColumn, array, offset, offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, rows,
                bottomRightColumn, array, offset, offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                -bottomRightColumn, array, offset, offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                columns, array, offset, offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(bottomRightRow + 1, topLeftColumn,
                topLeftRow, bottomRightColumn, array, offset, offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow,
                bottomRightColumn + 1, bottomRightRow, topLeftColumn, array, offset,
                offset + length - 1, false));
        assertThrows(IllegalArgumentException.class, () -> m.setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow,
                bottomRightColumn, array, offset, offset + length, false));
    }

    @Test
    void testIdentity() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        final var m = Matrix.identity(rows, columns);

        // check correctness
        assertEquals(m.getRows(), rows);
        assertEquals(m.getColumns(), columns);

        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                if (i == j) {
                    assertEquals(1.0, m.getElementAt(i, j), 0.0);
                } else {
                    assertEquals(0.0, m.getElementAt(i, j), 0.0);
                }
            }
        }

        // force WrongSizeException
        assertThrows(WrongSizeException.class, () -> Matrix.identity(0, columns));
        assertThrows(WrongSizeException.class, () -> Matrix.identity(rows, 0));
        assertThrows(WrongSizeException.class, () -> Matrix.identity(0, 0));
    }

    @Test
    void testFillWithUniformRandomValues() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        Matrix m;
        double value;
        var sum = 0.0;
        var sqrSum = 0.0;
        for (var k = 0; k < TIMES; k++) {
            m = new Matrix(rows, columns);
            Matrix.fillWithUniformRandomValues(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE, m);

            // check correctness
            assertEquals(m.getRows(), rows);
            assertEquals(m.getColumns(), columns);

            for (var j = 0; j < columns; j++) {
                for (var i = 0; i < rows; i++) {
                    value = m.getElementAt(i, j);

                    assertTrue(value >= MIN_RANDOM_VALUE);
                    assertTrue(value <= MAX_RANDOM_VALUE);

                    sum += value;
                    sqrSum += value * value;
                }
            }
        }

        final var numSamples = rows * columns * TIMES;
        final var estimatedMeanValue = sum / (double) (numSamples);
        final var estimatedVariance = (sqrSum - (double) numSamples *
                estimatedMeanValue * estimatedMeanValue) / ((double) numSamples - 1.0);

        // mean and variance of uniform distribution
        final var meanValue = 0.5 * (MIN_RANDOM_VALUE + MAX_RANDOM_VALUE);
        final var variance = (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) * (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) / 12.0;

        // check correctness of results
        assertEquals(estimatedMeanValue, meanValue, estimatedMeanValue * RELATIVE_ERROR);
        assertEquals(estimatedVariance, variance, estimatedVariance * RELATIVE_ERROR);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> Matrix.createWithUniformRandomValues(0, columns,
                MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        assertThrows(WrongSizeException.class, () -> Matrix.createWithUniformRandomValues(rows, 0,
                MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> Matrix.createWithUniformRandomValues(rows, columns,
                MAX_RANDOM_VALUE, MIN_RANDOM_VALUE));
    }

    @Test
    void testCreateWithUniformRandomValues() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        Matrix m;
        double value;
        var sum = 0.0;
        var sqrSum = 0.0;
        for (var k = 0; k < TIMES; k++) {
            m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
            // check correctness
            assertEquals(rows, m.getRows());
            assertEquals(columns, m.getColumns());

            for (var j = 0; j < columns; j++) {
                for (var i = 0; i < rows; i++) {
                    value = m.getElementAt(i, j);

                    assertTrue(value >= MIN_RANDOM_VALUE);
                    assertTrue(value <= MAX_RANDOM_VALUE);

                    sum += value;
                    sqrSum += value * value;
                }
            }
        }

        final var numSamples = rows * columns * TIMES;
        final var estimatedMeanValue = sum / (double) (numSamples);
        final var estimatedVariance = (sqrSum - (double) numSamples *
                estimatedMeanValue * estimatedMeanValue) / ((double) numSamples - 1.0);

        // mean and variance of uniform distribution
        final var meanValue = 0.5 * (MIN_RANDOM_VALUE + MAX_RANDOM_VALUE);
        final var variance = (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) * (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) / 12.0;

        // check correctness of results
        assertEquals(estimatedMeanValue, meanValue, estimatedMeanValue * RELATIVE_ERROR);
        assertEquals(estimatedVariance, variance, estimatedVariance * RELATIVE_ERROR);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> Matrix.createWithUniformRandomValues(0, columns,
                MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        assertThrows(WrongSizeException.class, () -> Matrix.createWithUniformRandomValues(rows, 0,
                MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> Matrix.createWithUniformRandomValues(rows, columns,
                MAX_RANDOM_VALUE, MIN_RANDOM_VALUE));
    }

    @Test
    void testCreateWithUniformRandomValues2() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        Matrix m;
        double value;
        var sum = 0.0;
        var sqrSum = 0.0;
        for (var k = 0; k < TIMES; k++) {
            m = Matrix.createWithUniformRandomValues(rows, columns, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE,
                    new SecureRandom());
            // check correctness
            assertEquals(rows, m.getRows());
            assertEquals(columns, m.getColumns());

            for (var j = 0; j < columns; j++) {
                for (var i = 0; i < rows; i++) {
                    value = m.getElementAt(i, j);

                    assertTrue(value >= MIN_RANDOM_VALUE);
                    assertTrue(value <= MAX_RANDOM_VALUE);

                    sum += value;
                    sqrSum += value * value;
                }
            }
        }

        final var numSamples = rows * columns * TIMES;
        final var estimatedMeanValue = sum / (double) (numSamples);
        final var estimatedVariance = (sqrSum - (double) numSamples * estimatedMeanValue * estimatedMeanValue)
                / ((double) numSamples - 1.0);

        // mean and variance of uniform distribution
        final var meanValue = 0.5 * (MIN_RANDOM_VALUE + MAX_RANDOM_VALUE);
        final var variance = (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) * (MAX_RANDOM_VALUE - MIN_RANDOM_VALUE) / 12.0;

        // check correctness of results
        assertEquals(estimatedMeanValue, meanValue, estimatedMeanValue * RELATIVE_ERROR);
        assertEquals(estimatedVariance, variance, estimatedVariance * RELATIVE_ERROR);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> Matrix.createWithUniformRandomValues(0, columns,
                MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        assertThrows(WrongSizeException.class, () -> Matrix.createWithUniformRandomValues(rows, 0,
                MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));

        //Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> Matrix.createWithUniformRandomValues(rows, columns,
                MAX_RANDOM_VALUE, MIN_RANDOM_VALUE));
    }

    @Test
    void testFillWithGaussianRandomValues() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        Matrix m;
        double value;
        var mean = 0.0;
        var sqrSum = 0.0;
        final var numSamples = rows * columns * TIMES;
        for (var k = 0; k < TIMES; k++) {
            m = new Matrix(rows, columns);
            Matrix.fillWithGaussianRandomValues(MEAN, STANDARD_DEVIATION, m);

            // check correctness
            assertEquals(rows, m.getRows());
            assertEquals(columns, m.getColumns());

            for (var j = 0; j < columns; j++) {
                for (var i = 0; i < rows; i++) {
                    value = m.getElementAt(i, j);

                    mean += value / (double) numSamples;
                    sqrSum += value * value / (double) numSamples;
                }
            }
        }

        final var standardDeviation = Math.sqrt(sqrSum - mean);

        // check correctness of results
        assertEquals(MEAN, mean, mean * RELATIVE_ERROR);
        assertEquals(STANDARD_DEVIATION, standardDeviation, standardDeviation * RELATIVE_ERROR);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> Matrix.createWithGaussianRandomValues(0, columns, MEAN,
                STANDARD_DEVIATION));
        assertThrows(WrongSizeException.class, () -> Matrix.createWithGaussianRandomValues(rows, 0, MEAN,
                STANDARD_DEVIATION));
        assertThrows(IllegalArgumentException.class, () -> Matrix.createWithGaussianRandomValues(rows, columns,
                MEAN, -STANDARD_DEVIATION));
    }

    @Test
    void testCreateWithGaussianRandomValues() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        Matrix m;
        double value;
        var mean = 0.0;
        var sqrSum = 0.0;
        final var numSamples = rows * columns * TIMES;
        for (var k = 0; k < TIMES; k++) {
            m = Matrix.createWithGaussianRandomValues(rows, columns, MEAN, STANDARD_DEVIATION);
            // check correctness
            assertEquals(rows, m.getRows());
            assertEquals(columns, m.getColumns());

            for (var j = 0; j < columns; j++) {
                for (var i = 0; i < rows; i++) {
                    value = m.getElementAt(i, j);

                    mean += value / (double) numSamples;
                    sqrSum += value * value / (double) numSamples;
                }
            }
        }

        final var standardDeviation = Math.sqrt(sqrSum - mean);

        // check correctness of results
        assertEquals(MEAN, mean, mean * RELATIVE_ERROR);
        assertEquals(STANDARD_DEVIATION, standardDeviation, standardDeviation * RELATIVE_ERROR);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> Matrix.createWithGaussianRandomValues(0, columns, MEAN,
                STANDARD_DEVIATION));
        assertThrows(WrongSizeException.class, () -> Matrix.createWithGaussianRandomValues(rows, 0, MEAN,
                STANDARD_DEVIATION));
        assertThrows(IllegalArgumentException.class, () -> Matrix.createWithGaussianRandomValues(rows, columns, MEAN,
                -STANDARD_DEVIATION));
    }

    @Test
    void testDiagonal() {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_ROWS, MAX_ROWS);

        final var diagonal = new double[length];

        // fill diagonal with random values
        for (var i = 0; i < length; i++) {
            diagonal[i] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        }

        final var m = Matrix.diagonal(diagonal);

        // check correctness
        assertEquals(length, m.getRows());
        assertEquals(length, m.getColumns());

        for (var j = 0; j < length; j++) {
            for (var i = 0; i < length; i++) {
                if (i == j) {
                    assertEquals(diagonal[i], m.getElementAt(i, j), 0.0);
                } else {
                    assertEquals(0.0, m.getElementAt(i, j), 0.0);
                }
            }
        }
    }

    @Test
    void testNewFromArray() {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var cols = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var length = rows * cols;

        final var array = new double[length];
        randomizer.fill(array, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        // use default column order
        var m = Matrix.newFromArray(array);
        var array2 = m.toArray();

        // check correctness
        assertEquals(length, m.getRows());
        assertEquals(1, m.getColumns());
        assertArrayEquals(array2, array, 0.0);

        // use column order
        m = Matrix.newFromArray(array, true);
        array2 = m.toArray(true);

        // check correctness
        assertEquals(length, m.getRows());
        assertEquals(1, m.getColumns());
        assertArrayEquals(array2, array, 0.0);

        // use row order
        m = Matrix.newFromArray(array, false);
        array2 = m.toArray(false);

        // check correctness
        assertEquals(1, m.getRows());
        assertEquals(length, m.getColumns());
        assertArrayEquals(array2, array, 0.0);
    }

    @Test
    void testFromArray() throws AlgebraException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var cols = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);
        final var length = rows * cols;

        final var array = new double[length];
        randomizer.fill(array, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var m = new Matrix(rows, cols);

        // use default column order
        m.fromArray(array);
        var array2 = m.toArray();

        // check correctness
        assertEquals(rows, m.getRows());
        assertEquals(cols, m.getColumns());
        assertArrayEquals(array2, array, 0.0);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> m.fromArray(new double[length + 1]));

        // use column order
        m.fromArray(array, true);
        array2 = m.toArray(true);

        // check correctness
        assertEquals(rows, m.getRows());
        assertEquals(cols, m.getColumns());
        assertArrayEquals(array2, array, 0.0);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> m.fromArray(new double[length + 1], true));

        // use row order
        m.fromArray(array, false);
        array2 = m.toArray(false);

        // check correctness
        assertEquals(rows, m.getRows());
        assertEquals(cols, m.getColumns());
        assertArrayEquals(array2, array, 0.0);

        // Force WrongSizeException
        assertThrows(WrongSizeException.class, () -> m.fromArray(new double[length + 1], false));
    }

    @Test
    void testSymmetrize() throws AlgebraException {
        var numValid = 0;
        for (var t = 0; t < TIMES; t++) {
            final var randomizer = new UniformRandomizer();
            final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);

            final var symmetric = DecomposerHelper.getSymmetricMatrix(rows);

            final var nonSymmetric = new Matrix(rows, rows);
            nonSymmetric.copyFrom(symmetric);
            nonSymmetric.setElementAt(0, rows - 1,
                    nonSymmetric.getElementAt(0, rows - 1) + 1.0);

            // symmetrize
            final var symmetric2 = new Matrix(rows, rows);
            symmetric.symmetrize(symmetric2);

            final var nonSymmetric2 = new Matrix(rows, rows);
            nonSymmetric.symmetrize(nonSymmetric2);

            // check correctness
            if (!Utils.isSymmetric(symmetric)) {
                continue;
            }
            assertTrue(Utils.isSymmetric(symmetric));
            if (Utils.isSymmetric(nonSymmetric)) {
                continue;
            }
            assertFalse(Utils.isSymmetric(nonSymmetric));

            if (!Utils.isSymmetric(symmetric2)) {
                continue;
            }
            assertTrue(Utils.isSymmetric(symmetric2));
            if (!Utils.isSymmetric(nonSymmetric2)) {
                continue;
            }
            assertTrue(Utils.isSymmetric(nonSymmetric2));

            var failed = false;
            for (var i = 0; i < symmetric2.getColumns(); i++) {
                for (var j = 0; j < symmetric2.getRows(); j++) {
                    if (Math.abs(symmetric2.getElementAt(i, j) - 0.5 * (symmetric.getElementAt(j, i)
                            + symmetric.getElementAt(j, i))) > ABSOLUTE_ERROR) {
                        failed = true;
                        break;
                    }
                    assertEquals(0.5 * (symmetric.getElementAt(i, j) + symmetric.getElementAt(j, i)),
                            symmetric2.getElementAt(i, j), ABSOLUTE_ERROR);
                }
            }

            if (failed) {
                continue;
            }

            for (var i = 0; i < nonSymmetric2.getColumns(); i++) {
                for (var j = 0; j < nonSymmetric2.getRows(); j++) {
                    if (Math.abs(nonSymmetric2.getElementAt(i, j) - 0.5 * (nonSymmetric.getElementAt(i, j)
                            + nonSymmetric.getElementAt(j, i))) > ABSOLUTE_ERROR) {
                        failed = true;
                        break;
                    }
                    assertEquals(0.5 * (nonSymmetric.getElementAt(i, j) + nonSymmetric.getElementAt(j, i)),
                            nonSymmetric2.getElementAt(i, j), ABSOLUTE_ERROR);
                }
            }

            if (failed) {
                continue;
            }


            // Force WrongSizeException
            final var wrong = new Matrix(1, 2);
            assertThrows(WrongSizeException.class, () -> wrong.symmetrize(wrong));
            assertThrows(WrongSizeException.class, () -> symmetric.symmetrize(wrong));

            // symmetrize and return new
            final var symmetric3 = symmetric.symmetrizeAndReturnNew();
            final var nonSymmetric3 = nonSymmetric.symmetrizeAndReturnNew();

            // check correctness
            if (!Utils.isSymmetric(symmetric)) {
                continue;
            }
            assertTrue(Utils.isSymmetric(symmetric));
            if (Utils.isSymmetric(nonSymmetric)) {
                continue;
            }
            assertFalse(Utils.isSymmetric(nonSymmetric));

            if (!Utils.isSymmetric(symmetric3)) {
                continue;
            }
            assertTrue(Utils.isSymmetric(symmetric3));
            if (!Utils.isSymmetric(nonSymmetric3)) {
                continue;
            }
            assertTrue(Utils.isSymmetric(nonSymmetric3));

            // Force WrongSizeException
            assertThrows(WrongSizeException.class, wrong::symmetrizeAndReturnNew);

            // symmetrize and update
            symmetric.symmetrize();
            nonSymmetric.symmetrize();

            // check correctness
            if (!Utils.isSymmetric(symmetric)) {
                continue;
            }
            assertTrue(Utils.isSymmetric(symmetric));
            if (!Utils.isSymmetric(nonSymmetric)) {
                continue;
            }
            assertTrue(Utils.isSymmetric(nonSymmetric));

            // Force WrongSizeException
            assertThrows(WrongSizeException.class, wrong::symmetrize);

            numValid++;
            break;
        }

        assertTrue(numValid > 0);
    }

    @Test
    void testSerializeDeserialize() throws WrongSizeException, IOException, ClassNotFoundException {
        final var randomizer = new UniformRandomizer();
        final var rows = randomizer.nextInt(MIN_ROWS, MAX_ROWS);
        final var columns = randomizer.nextInt(MIN_COLUMNS, MAX_COLUMNS);

        // instantiate matrix and fill with random values
        final var m1 = new Matrix(rows, columns);
        for (var j = 0; j < columns; j++) {
            for (var i = 0; i < rows; i++) {
                m1.setElementAt(i, j, randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            }
        }

        final var bytes = SerializationHelper.serialize(m1);
        final var m2 = SerializationHelper.deserialize(bytes);

        assertEquals(m2, m1);
        assertNotSame(m2, m1);
    }
}
