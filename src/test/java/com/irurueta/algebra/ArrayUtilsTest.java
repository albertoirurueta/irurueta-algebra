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

class ArrayUtilsTest {

    private static final int MIN_LENGTH = 1;
    private static final int MAX_LENGTH = 50;

    private static final double MIN_RANDOM_VALUE = -100.0;
    private static final double MAX_RANDOM_VALUE = 100.0;

    private static final double ABSOLUTE_ERROR = 1e-6;

    private static final int TIMES = 100;

    @Test
    void testMultiplyByScalar() {

        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);
        final var scalar = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var input = new double[length];
        randomizer.fill(input, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var expectedResult = new double[length];
        for (int i = 0; i < length; i++) {
            expectedResult[i] = input[i] * scalar;
        }

        final var result1 = ArrayUtils.multiplyByScalarAndReturnNew(input, scalar);

        final var result2 = new double[length];
        ArrayUtils.multiplyByScalar(input, scalar, result2);

        // check correctness
        assertEquals(result1.length, length);
        for (int i = 0; i < length; i++) {
            assertEquals(expectedResult[i], result1[i], 0.0);
            assertEquals(expectedResult[i], result2[i], 0.0);
        }

        // Force IllegalArgumentException
        final var wrongResult = new double[length + 1];
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.multiplyByScalar(input, scalar, wrongResult));
    }

    @Test
    void testSum() {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        final var input1 = new double[length];
        final var input2 = new double[length];
        randomizer.fill(input1, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        randomizer.fill(input2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var expectedResult = new double[length];
        for (var i = 0; i < length; i++) {
            expectedResult[i] = input1[i] + input2[i];
        }

        final var result1 = ArrayUtils.sumAndReturnNew(input1, input2);

        final var result2 = new double[length];
        ArrayUtils.sum(input1, input2, result2);

        // check correctness
        assertEquals(result1.length, length);
        for (var i = 0; i < length; i++) {
            assertEquals(expectedResult[i], result1[i], 0.0);
            assertEquals(expectedResult[i], result2[i], 0.0);
        }

        // Force IllegalArgumentException
        final var wrongArray = new double[length + 1];
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.sum(input1, input2, wrongArray));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.sumAndReturnNew(input1, wrongArray));
    }

    @Test
    void testSubtract() {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        final var input1 = new double[length];
        final var input2 = new double[length];
        randomizer.fill(input1, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        randomizer.fill(input2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var expectedResult = new double[length];
        for (int i = 0; i < length; i++) {
            expectedResult[i] = input1[i] - input2[i];
        }

        final var result1 = ArrayUtils.subtractAndReturnNew(input1, input2);

        final var result2 = new double[length];
        ArrayUtils.subtract(input1, input2, result2);

        // check correctness
        assertEquals(result1.length, length);
        for (var i = 0; i < length; i++) {
            assertEquals(expectedResult[i], result1[i], 0.0);
            assertEquals(expectedResult[i], result2[i], 0.0);
        }

        // Force IllegalArgumentException
        final var wrongArray = new double[length + 1];
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.subtract(input1, input2, wrongArray));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.subtractAndReturnNew(input1, wrongArray));
    }

    @Test
    void testDotProduct() throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH + 1, MAX_LENGTH);

        final var input1 = new double[length];
        final var input2 = new double[length];
        randomizer.fill(input1, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        randomizer.fill(input2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        var expectedResult = 0.0;
        for (int i = 0; i < length; i++) {
            expectedResult += input1[i] * input2[i];
        }

        var result = ArrayUtils.dotProduct(input1, input2);

        // check correctness
        assertEquals(expectedResult, result, 0.0);

        // Force IllegalArgumentException
        final var wrongArray = new double[length + 1];
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.dotProduct(input1, wrongArray));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.dotProduct(wrongArray, input2));

        // Test with jacobians
        final var jacobian1 = new Matrix(1, length);
        final var jacobian2 = new Matrix(1, length);
        result = ArrayUtils.dotProduct(input1, input2, jacobian1, jacobian2);

        // check correctness
        assertEquals(expectedResult, result, 0.0);

        assertArrayEquals(input1, jacobian1.getBuffer(), 0.0);
        assertArrayEquals(input2, jacobian2.getBuffer(), 0.0);

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class,
                () -> ArrayUtils.dotProduct(wrongArray, input2, jacobian1, jacobian2));
        assertThrows(IllegalArgumentException.class,
                () -> ArrayUtils.dotProduct(input1, wrongArray, jacobian1, jacobian2));
        assertThrows(IllegalArgumentException.class,
                () -> ArrayUtils.dotProduct(input1, input2, new Matrix(1, 1), jacobian2));
        assertThrows(IllegalArgumentException.class,
                () -> ArrayUtils.dotProduct(input1, input2, jacobian1, new Matrix(1, 1)));
    }

    @Test
    void testAngle() {
        var numValid = 0;
        for (var t = 0; t < TIMES; t++) {
            final var randomizer = new UniformRandomizer();
            final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

            final var input1 = new double[length];
            final var input2 = new double[length];
            randomizer.fill(input1, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
            randomizer.fill(input2, MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

            final var norm1 = Utils.normF(input1);
            final var norm2 = Utils.normF(input2);

            if (norm1 < ABSOLUTE_ERROR || norm2 < ABSOLUTE_ERROR) {
                continue;
            }

            var dotProduct = 0.0;
            for (var i = 0; i < length; i++) {
                dotProduct += input1[i] * input2[i];
            }

            final var expectedResult = Math.acos(dotProduct / norm1 / norm2);

            final var result = ArrayUtils.angle(input1, input2);

            // check correctness
            assertEquals(expectedResult, result, ABSOLUTE_ERROR);
            assertEquals(expectedResult, ArrayUtils.angle(input2, input1), ABSOLUTE_ERROR);
            assertEquals(0.0, ArrayUtils.angle(input1, input1), ABSOLUTE_ERROR);
            assertEquals(0.0, ArrayUtils.angle(input2, input2), ABSOLUTE_ERROR);

            numValid++;
            break;
        }

        assertTrue(numValid > 0);
    }

    @Test
    void testMultiplyByScalarComplex() {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);
        final var scalar = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var input = new Complex[length];
        // fill array with random values
        for (var i = 0; i < length; i++) {
            input[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE,
                    MAX_RANDOM_VALUE), randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        }

        final var expectedResult = new Complex[length];
        for (var i = 0; i < length; i++) {
            expectedResult[i] = input[i].multiplyByScalarAndReturnNew(scalar);
        }

        final var result1 = ArrayUtils.multiplyByScalarAndReturnNew(input, scalar);

        final var result2 = new Complex[length];
        // Force NullPointerException (because result2 hasn't been initialized
        // with instances in the array
        assertThrows(NullPointerException.class, () -> ArrayUtils.multiplyByScalar(input, scalar, result2));

        // initialize array with instances (otherwise null pointer exception will
        // be raised
        for (var i = 0; i < length; i++) {
            result2[i] = new Complex();
        }
        ArrayUtils.multiplyByScalar(input, scalar, result2);

        // check correctness
        assertEquals(result1.length, length);
        for (var i = 0; i < length; i++) {
            assertEquals(expectedResult[i], result1[i]);
            assertEquals(expectedResult[i], result2[i]);
        }

        // Force IllegalArgumentException
        final var wrongResult = new Complex[length + 1];
        for (var i = 0; i < length; i++) {
            wrongResult[i] = new Complex();
        }
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.multiplyByScalar(input, scalar, wrongResult));
    }

    @Test
    void testSumComplex() {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        final var input1 = new Complex[length];
        final var input2 = new Complex[length];
        // fill array with random values
        for (var i = 0; i < length; i++) {
            input1[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            input2[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        }

        final var expectedResult = new Complex[length];
        for (int i = 0; i < length; i++) {
            expectedResult[i] = input1[i].addAndReturnNew(input2[i]);
        }

        final var result1 = ArrayUtils.sumAndReturnNew(input1, input2);

        final var result2 = new Complex[length];
        // Force NullPointerException (because result2 hasn't been initialized
        // with instances in the array)
        assertThrows(NullPointerException.class, () -> ArrayUtils.sum(input1, input2, result2));

        // initialize array with instances (otherwise null pointer exception will
        // be raised
        for (var i = 0; i < length; i++) {
            result2[i] = new Complex();
        }
        ArrayUtils.sum(input1, input2, result2);

        // check correctness
        assertEquals(result1.length, length);
        for (var i = 0; i < length; i++) {
            assertEquals(result1[i], expectedResult[i]);
            assertEquals(result2[i], expectedResult[i]);
        }

        // Force IllegalArgumentException
        final var wrongArray = new Complex[length + 1];
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.sum(input1, input2, wrongArray));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.sumAndReturnNew(input1, wrongArray));
    }

    @Test
    void testSubtractComplex() {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        final var input1 = new Complex[length];
        final var input2 = new Complex[length];
        // fill array with random values
        for (var i = 0; i < length; i++) {
            input1[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            input2[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        }

        final var expectedResult = new Complex[length];
        for (var i = 0; i < length; i++) {
            expectedResult[i] = input1[i].subtractAndReturnNew(input2[i]);
        }

        final var result1 = ArrayUtils.subtractAndReturnNew(input1, input2);

        final var result2 = new Complex[length];
        // Force NullPointerException (because result2 hasn't been initialized
        // with instances in the array)
        assertThrows(NullPointerException.class, () -> ArrayUtils.subtract(input1, input2, result2));

        // initialize array with instances (otherwise null pointer exception will
        // be raised
        for (var i = 0; i < length; i++) {
            result2[i] = new Complex();
        }
        ArrayUtils.subtract(input1, input2, result2);

        // check correctness
        assertEquals(result1.length, length);
        for (var i = 0; i < length; i++) {
            assertEquals(expectedResult[i], result1[i]);
            assertEquals(expectedResult[i], result2[i]);
        }

        // Force IllegalArgumentException
        final var wrongArray = new Complex[length + 1];
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.subtract(input1, input2, wrongArray));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.subtractAndReturnNew(input1, wrongArray));
    }

    @Test
    void testDotProductComplex() {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        final var input1 = new Complex[length];
        final var input2 = new Complex[length];
        // fill array with random values
        for (var i = 0; i < length; i++) {
            input1[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
            input2[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        }

        final var expectedResult = new Complex(0.0);
        for (var i = 0; i < length; i++) {
            expectedResult.add(input1[i].multiplyAndReturnNew(input2[i]));
        }

        final var result = ArrayUtils.dotProduct(input1, input2);

        // check correctness
        assertEquals(expectedResult, result);

        // Force IllegalArgumentException
        final var wrongArray = new Complex[length + 1];
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.dotProduct(input1, wrongArray));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.dotProduct(wrongArray, input2));
    }

    @Test
    void testNormalize() throws AlgebraException {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        var v = new double[length];
        randomizer.fill(v);

        var result = new double[length];
        var result1 = new double[length];
        var jacobian = new Matrix(length, length);

        ArrayUtils.normalize(v, result, jacobian);
        ArrayUtils.normalize(v, result1);

        // check correctness
        final var norm = Utils.normF(v);
        final var result2 = ArrayUtils.multiplyByScalarAndReturnNew(v, 1.0 / norm);
        assertArrayEquals(result, result2, ABSOLUTE_ERROR);
        assertArrayEquals(result1, result2, 0.0);

        final var jacobian2 = Matrix.identity(length, length);
        jacobian2.multiplyByScalar(norm * norm);
        jacobian2.subtract(Matrix.newFromArray(v, true).multiplyAndReturnNew(
                Matrix.newFromArray(v, false)));
        jacobian2.multiplyByScalar(1.0 / (norm * norm * norm));

        assertTrue(jacobian.equals(jacobian2, ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        final var otherJacobian = new Matrix(length, length);
        final var otherResult = new double[length];
        final var otherV = new double[length];
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.normalize(new double[length + 1], otherResult,
                otherJacobian));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.normalize(otherV, new double[length + 1],
                otherJacobian));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.normalize(otherV, otherResult,
                new Matrix(length + 1, length)));

        // test normalize and return new with jacobian
        jacobian = new Matrix(length, length);
        result = ArrayUtils.normalizeAndReturnNew(v, jacobian);

        // check correctness
        assertArrayEquals(result, result2, ABSOLUTE_ERROR);
        assertTrue(jacobian.equals(jacobian2, ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.normalizeAndReturnNew(new double[length + 1],
                otherJacobian));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.normalizeAndReturnNew(otherV,
                new Matrix(length + 1, length)));

        // test normalize and update with jacobian
        var v2 = Arrays.copyOf(v, length);
        jacobian = new Matrix(length, length);
        ArrayUtils.normalize(v2, jacobian);

        // check correctness
        assertArrayEquals(result, v2, ABSOLUTE_ERROR);
        assertTrue(jacobian.equals(jacobian2, ABSOLUTE_ERROR));

        // Force IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.normalize(new double[length + 1], otherJacobian));
        assertThrows(IllegalArgumentException.class, () -> ArrayUtils.normalize(otherV,
                new Matrix(length + 1, length)));

        // test normalize and return new
        result = ArrayUtils.normalizeAndReturnNew(v);

        // check correctness
        assertArrayEquals(result2, result, ABSOLUTE_ERROR);

        // test normalize and update
        v2 = Arrays.copyOf(v, length);
        ArrayUtils.normalize(v2);

        // check correctness
        assertArrayEquals(v2, result, ABSOLUTE_ERROR);


        // test for zero norm
        v = new double[length];
        result = new double[length];
        jacobian = new Matrix(length, length);

        ArrayUtils.normalize(v, result, jacobian);

        // check
        Arrays.fill(result2, Double.MAX_VALUE);
        assertArrayEquals(result2, result, 0.0);

        for (var i = 0; i < length; i++) {
            for (var j = 0; j < length; j++) {
                assertEquals(Double.MAX_VALUE, jacobian.getElementAt(i, j), 0.0);
            }
        }
    }

    @Test
    void testReverse() {
        final var randomizer = new UniformRandomizer();
        var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        var v = new double[length];
        randomizer.fill(v);

        var result = new double[length];
        ArrayUtils.reverse(v, result);

        // check correctness
        var result2 = new double[length];
        for (var i = 0; i < length; i++) {
            result2[length - 1 - i] = v[i];
        }

        assertArrayEquals(result2, result, ABSOLUTE_ERROR);

        ArrayUtils.reverse(v);

        // check correctness
        assertArrayEquals(result2, v, ABSOLUTE_ERROR);

        // test for odd/even case
        length++; // if length was even it will be now odd, and vice versa

        v = new double[length];
        randomizer.fill(v);

        result = new double[length];
        ArrayUtils.reverse(v, result);

        // check correctness
        result2 = new double[length];
        for (var i = 0; i < length; i++) {
            result2[length - 1 - i] = v[i];
        }

        assertArrayEquals(result2, result, ABSOLUTE_ERROR);

        ArrayUtils.reverse(v);

        // check correctness
        assertArrayEquals(result2, v, ABSOLUTE_ERROR);
    }

    @Test
    void testReverseAndReturnNew() {
        final var randomizer = new UniformRandomizer();
        var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        var v = new double[length];
        randomizer.fill(v);

        var result = ArrayUtils.reverseAndReturnNew(v);

        // check correctness
        var result2 = new double[length];
        for (var i = 0; i < length; i++) {
            result2[length - 1 - i] = v[i];
        }

        assertArrayEquals(result2, result, ABSOLUTE_ERROR);

        // test for odd/even case
        length++; // if length was even it will be now odd, and vice versa

        v = new double[length];
        randomizer.fill(v);

        result = ArrayUtils.reverseAndReturnNew(v);

        // check correctness
        result2 = new double[length];
        for (var i = 0; i < length; i++) {
            result2[length - 1 - i] = v[i];
        }

        assertArrayEquals(result2, result, ABSOLUTE_ERROR);
    }

    @Test
    void testReverseComplex() {
        final var randomizer = new UniformRandomizer();
        var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        var v = new Complex[length];
        // fill array with random values
        for (var i = 0; i < length; i++) {
            v[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        }

        var result = new Complex[length];
        ArrayUtils.reverse(v, result);

        // check correctness
        assertEquals(result.length, length);
        for (var i = 0; i < length; i++) {
            assertEquals(v[i], result[length - 1 - i]);
        }

        // copy and reverse
        var result2 = new Complex[length];
        for (var i = 0; i < length; i++) {
            result2[i] = new Complex(result[i]);
        }
        ArrayUtils.reverse(v);

        // check correctness
        assertEquals(v.length, result2.length);
        for (var i = 0; i < length; i++) {
            assertEquals(result2[i], v[i]);
        }

        // test for odd/even case
        length++; // if length was even it will be now odd, and vice versa

        v = new Complex[length];
        // fill array with random values
        for (var i = 0; i < length; i++) {
            v[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        }

        result = new Complex[length];
        ArrayUtils.reverse(v, result);

        // check correctness
        assertEquals(result.length, length);
        for (var i = 0; i < length; i++) {
            assertEquals(v[i], result[length - 1 - i]);
        }

        // copy and reverse
        result2 = new Complex[length];
        for (var i = 0; i < length; i++) {
            result2[i] = new Complex(result[i]);
        }
        ArrayUtils.reverse(v);

        // check correctness
        assertEquals(v.length, result2.length);
        for (var i = 0; i < length; i++) {
            assertEquals(result2[i], v[i]);
        }
    }

    @Test
    void testReverseAndReturnNewComplex() {
        final var randomizer = new UniformRandomizer();
        var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        var v = new Complex[length];
        // fill array with random values
        for (var i = 0; i < length; i++) {
            v[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        }

        var result = ArrayUtils.reverseAndReturnNew(v);

        // check correctness
        assertEquals(length, result.length);
        for (var i = 0; i < length; i++) {
            assertEquals(v[i], result[length - 1 - i]);
        }

        // test for odd/even case
        length++; // if length was even it will be now odd, and vice versa

        v = new Complex[length];
        // fill array with random values
        for (var i = 0; i < length; i++) {
            v[i] = new Complex(randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE),
                    randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE));
        }

        result = ArrayUtils.reverseAndReturnNew(v);

        // check correctness
        assertEquals(length, result.length);
        for (var i = 0; i < length; i++) {
            assertEquals(v[i], result[length - 1 - i]);
        }
    }

    @Test
    void testSqrt() {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        final var v = new double[length];
        final var sqrt = new double[length];
        for (var i = 0; i < length; i++) {
            v[i] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
            sqrt[i] = Math.sqrt(v[i]);
        }

        // check correctness
        var sqrt2 = new double[length];
        ArrayUtils.sqrt(v, sqrt2);

        assertArrayEquals(sqrt2, sqrt, ABSOLUTE_ERROR);

        sqrt2 = ArrayUtils.sqrtAndReturnNew(v);
        assertArrayEquals(sqrt2, sqrt, ABSOLUTE_ERROR);

        ArrayUtils.sqrt(v);
        assertArrayEquals(sqrt2, v, ABSOLUTE_ERROR);
    }

    @Test
    void testMinMax() {
        final var randomizer = new UniformRandomizer();
        final var length = randomizer.nextInt(MIN_LENGTH, MAX_LENGTH);

        final var v = new double[length];
        var minValue = Double.MAX_VALUE;
        var maxValue = -Double.MAX_VALUE;
        var minPos = -1;
        var maxPos = -1;
        for (var i = 0; i < length; i++) {
            v[i] = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
            if (v[i] < minValue) {
                minValue = v[i];
                minPos = i;
            }
            if (v[i] > maxValue) {
                maxValue = v[i];
                maxPos = i;
            }
        }

        var pos = new int[1];
        assertEquals(minValue, ArrayUtils.min(v, pos), 0.0);
        assertEquals(pos[0], minPos);

        assertEquals(minValue, ArrayUtils.min(v), 0.0);

        assertEquals(maxValue, ArrayUtils.max(v, pos), 0.0);
        assertEquals(pos[0], maxPos);

        assertEquals(maxValue, ArrayUtils.max(v), 0.0);

        final var result = new double[2];
        pos = new int[2];
        ArrayUtils.minMax(v, result, pos);
        assertEquals(minValue, result[0], 0.0);
        assertEquals(maxValue, result[1], 0.0);
        assertEquals(minPos, pos[0]);
        assertEquals(maxPos, pos[1]);
    }
}
