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

import static org.junit.jupiter.api.Assertions.*;

class ComplexTest {

    private static final double MIN_RANDOM_VALUE = -100.0;

    private static final double MAX_RANDOM_VALUE = 100.0;

    private static final double MIN_MODULUS = 1.0;

    private static final double MAX_MODULUS = 10.0;

    private static final double MIN_PHASE = -Math.PI;

    private static final double MAX_PHASE = Math.PI;

    private static final double MIN_EXPONENT = -2.0;

    private static final double MAX_EXPONENT = 2.0;

    private static final double ABSOLUTE_ERROR = 1e-9;

    @Test
    void testConstructor() {
        Complex c;

        // Test 1st constructor
        c = new Complex();
        assertNotNull(c);
        assertEquals(0.0, c.getReal(), 0.0);
        assertEquals(0.0, c.getImaginary(), 0.0);
        assertEquals(0.0, c.getModulus(), ABSOLUTE_ERROR);

        final var randomizer = new UniformRandomizer();
        final var real = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        // Test 2nd constructor
        c = new Complex(real);
        assertNotNull(c);
        assertEquals(real, c.getReal(), 0.0);
        assertEquals(0.0, c.getImaginary(), 0.0);
        assertEquals(Math.abs(real), c.getModulus(), ABSOLUTE_ERROR);

        // Test 3rd constructor
        c = new Complex(real, imaginary);
        assertNotNull(c);
        assertEquals(real, c.getReal(), 0.0);
        assertEquals(imaginary, c.getImaginary(), 0.0);
        assertEquals(Math.sqrt(real * real + imaginary * imaginary), c.getModulus(), ABSOLUTE_ERROR);
        assertEquals(Math.atan2(imaginary, real), c.getPhase(), ABSOLUTE_ERROR);

        // Test 4th constructor
        var c2 = new Complex(c);
        assertEquals(c.getReal(), c2.getReal(), 0.0);
        assertEquals(c.getImaginary(), c2.getImaginary(), 0.0);
        assertEquals(real, c2.getReal(), 0.0);
        assertEquals(imaginary, c2.getImaginary(), 0.0);
        assertEquals(c.getModulus(), c2.getModulus(), 0.0);
        assertEquals(c.getPhase(), c2.getPhase(), 0.0);
    }

    @Test
    void testGetSetReal() {
        final var randomizer = new UniformRandomizer();
        final var real = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var c = new Complex();

        assertEquals(0.0, c.getReal(), 0.0);

        // set new value
        c.setReal(real);
        // check correctness
        assertEquals(real, c.getReal(), 0.0);
    }

    @Test
    void testGetSetImaginary() {
        final var randomizer = new UniformRandomizer();
        final var imaginary = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var c = new Complex();

        assertEquals(0.0, c.getImaginary(), 0.0);

        // set new value
        c.setImaginary(imaginary);
        // check correctness
        assertEquals(imaginary, c.getImaginary(), 0.0);
    }

    @Test
    void testGetSetRealAndImaginary() {
        final var randomizer = new UniformRandomizer();
        final var real = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var c = new Complex();

        assertEquals(0.0, c.getReal(), 0.0);
        assertEquals(0.0, c.getImaginary(), 0.0);

        // set new value
        c.setRealAndImaginary(real, imaginary);
        // check correctness
        assertEquals(real, c.getReal(), 0.0);
        assertEquals(imaginary, c.getImaginary(), 0.0);
    }

    @Test
    void testGetModulus() {
        final var randomizer = new UniformRandomizer();
        final var real = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var modulus = Math.sqrt(real * real + imaginary * imaginary);
        final var c = new Complex(real, imaginary);

        assertEquals(modulus, c.getModulus(), ABSOLUTE_ERROR);
    }

    @Test
    void testGetPhase() {
        final var randomizer = new UniformRandomizer();
        final var real = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var phase = Math.atan2(imaginary, real);
        final var c = new Complex(real, imaginary);

        assertEquals(phase, c.getPhase(), ABSOLUTE_ERROR);
    }

    @Test
    void testSetModulusAndPhase() {
        final var randomizer = new UniformRandomizer();
        final var modulus = randomizer.nextDouble(MIN_MODULUS, MAX_MODULUS);
        final var phase = randomizer.nextDouble(MIN_PHASE, MAX_PHASE);

        final var real = modulus * Math.cos(phase);
        final var imaginary = modulus * Math.sin(phase);
        final var c = new Complex();

        c.setModulusAndPhase(modulus, phase);
        assertEquals(modulus, c.getModulus(), ABSOLUTE_ERROR);
        assertEquals(phase, c.getPhase(), ABSOLUTE_ERROR);
        assertEquals(real, c.getReal(), ABSOLUTE_ERROR);
        assertEquals(imaginary, c.getImaginary(), ABSOLUTE_ERROR);
    }

    @Test
    void testConjugate() {
        final var randomizer = new UniformRandomizer();
        final var modulus = randomizer.nextDouble(MIN_MODULUS, MAX_MODULUS);
        final var phase = randomizer.nextDouble(MIN_PHASE, MAX_PHASE);

        final var real = modulus * Math.cos(phase);
        final var imaginary = modulus * Math.sin(phase);

        final var c = new Complex(real, imaginary);
        assertEquals(real, c.getReal(), 0.0);
        assertEquals(imaginary, c.getImaginary(), 0.0);

        var result = new Complex();
        assertEquals(0.0, result.getReal(), 0.0);
        assertEquals(0.0, result.getImaginary(), 0.0);

        // Test conjugate and store in result
        c.conjugate(result);
        assertEquals(real, result.getReal(), 0.0);
        assertEquals(-imaginary, result.getImaginary(), 0.0);
        assertEquals(modulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(-phase, result.getPhase(), ABSOLUTE_ERROR);

        // Test conjugate and return new
        result = c.conjugateAndReturnNew();
        assertEquals(real, result.getReal(), 0.0);
        assertEquals(-imaginary, result.getImaginary(), 0.0);
        assertEquals(modulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(-phase, result.getPhase(), ABSOLUTE_ERROR);

        // Test conjugate itself
        c.conjugate();
        assertEquals(real, c.getReal(), 0.0);
        assertEquals(-imaginary, c.getImaginary(), 0.0);
        assertEquals(modulus, c.getModulus(), ABSOLUTE_ERROR);
        assertEquals(-phase, c.getPhase(), ABSOLUTE_ERROR);
    }

    @Test
    void testAdd() {
        final var randomizer = new UniformRandomizer();
        final var real1 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary1 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var real2 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary2 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var c1 = new Complex(real1, imaginary1);
        final var c2 = new Complex(real2, imaginary2);
        var result = new Complex();

        assertEquals(real1, c1.getReal(), 0.0);
        assertEquals(imaginary1, c1.getImaginary(), 0.0);
        assertEquals(real2, c2.getReal(), 0.0);
        assertEquals(imaginary2, c2.getImaginary(), 0.0);
        assertEquals(0.0, result.getReal(), 0.0);
        assertEquals(0.0, result.getImaginary(), 0.0);

        // Add and store in result
        c1.add(c2, result);
        // check correctness
        assertEquals(real1 + real2, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(imaginary1 + imaginary2, result.getImaginary(), ABSOLUTE_ERROR);

        // Add and return result
        result = c1.addAndReturnNew(c2);
        // check correctness
        assertEquals(real1 + real2, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(imaginary1 + imaginary2, result.getImaginary(), ABSOLUTE_ERROR);

        // Add and store result on same instance
        c1.add(c2);
        // check correctness
        assertEquals(real1 + real2, c1.getReal(), ABSOLUTE_ERROR);
        assertEquals(imaginary1 + imaginary2, c1.getImaginary(), ABSOLUTE_ERROR);
    }

    @Test
    void testSubtract() {
        final var randomizer = new UniformRandomizer();
        final var real1 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary1 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var real2 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary2 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var c1 = new Complex(real1, imaginary1);
        final var c2 = new Complex(real2, imaginary2);
        var result = new Complex();

        assertEquals(real1, c1.getReal(), 0.0);
        assertEquals(imaginary1, c1.getImaginary(), 0.0);
        assertEquals(real2, c2.getReal(), 0.0);
        assertEquals(imaginary2, c2.getImaginary(), 0.0);
        assertEquals(0.0, result.getReal(), 0.0);
        assertEquals(0.0, result.getImaginary(), 0.0);

        // Subtract and store in result
        c1.subtract(c2, result);
        // check correctness
        assertEquals(real1 - real2, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(imaginary1 - imaginary2, result.getImaginary(), ABSOLUTE_ERROR);

        // Subtract and return result
        result = c1.subtractAndReturnNew(c2);
        // check correctness
        assertEquals(real1 - real2, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(imaginary1 - imaginary2, result.getImaginary(), ABSOLUTE_ERROR);

        // Subtract and store result on same instance
        c1.subtract(c2);
        // check correctness
        assertEquals(real1 - real2, c1.getReal(), ABSOLUTE_ERROR);
        assertEquals(imaginary1 - imaginary2, c1.getImaginary(), ABSOLUTE_ERROR);
    }

    @Test
    void testMultiply() {
        final var randomizer = new UniformRandomizer();
        final var modulus1 = randomizer.nextDouble(MIN_MODULUS, MAX_MODULUS);
        final var phase1 = randomizer.nextDouble(MIN_PHASE, MAX_PHASE);
        final var modulus2 = randomizer.nextDouble(MIN_MODULUS, MAX_MODULUS);
        final var phase2 = randomizer.nextDouble(MIN_PHASE, MAX_PHASE);

        final var real1 = modulus1 * Math.cos(phase1);
        final var imaginary1 = modulus1 * Math.sin(phase1);
        final var real2 = modulus2 * Math.cos(phase2);
        final var imaginary2 = modulus2 * Math.sin(phase2);

        final var c1 = new Complex(real1, imaginary1);
        final var c2 = new Complex(real2, imaginary2);
        var result = new Complex();

        assertEquals(real1, c1.getReal(), 0.0);
        assertEquals(imaginary1, c1.getImaginary(), 0.0);
        assertEquals(real2, c2.getReal(), 0.0);
        assertEquals(imaginary2, c2.getImaginary(), 0.0);
        assertEquals(0.0, result.getReal(), 0.0);
        assertEquals(0.0, result.getImaginary(), 0.0);

        // multiply and store in result
        c1.multiply(c2, result);
        // check correctness
        final var resultModulus = modulus1 * modulus2;
        var resultPhase = phase1 + phase2;
        final var resultReal = resultModulus * Math.cos(resultPhase);
        final var resultImaginary = resultModulus * Math.sin(resultPhase);
        resultPhase = Math.atan2(resultImaginary, resultReal);
        assertEquals(resultModulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, result.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // multiply and return result
        result = c1.multiplyAndReturnNew(c2);
        // check correctness
        assertEquals(resultModulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, result.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // multiply and store result on same instance
        c1.multiply(c2);
        // check correctness
        assertEquals(resultModulus, c1.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, c1.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, c1.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, c1.getImaginary(), ABSOLUTE_ERROR);
    }

    @Test
    void testDivide() {
        final var randomizer = new UniformRandomizer();
        final var modulus1 = randomizer.nextDouble(MIN_MODULUS, MAX_MODULUS);
        final var phase1 = randomizer.nextDouble(MIN_PHASE, MAX_PHASE);
        final var modulus2 = randomizer.nextDouble(MIN_MODULUS, MAX_MODULUS);
        final var phase2 = randomizer.nextDouble(MIN_PHASE, MAX_PHASE);

        final var real1 = modulus1 * Math.cos(phase1);
        final var imaginary1 = modulus1 * Math.sin(phase1);
        final var real2 = modulus2 * Math.cos(phase2);
        final var imaginary2 = modulus2 * Math.sin(phase2);

        final var c1 = new Complex(real1, imaginary1);
        final var c2 = new Complex(real2, imaginary2);
        var result = new Complex();

        assertEquals(real1, c1.getReal(), 0.0);
        assertEquals(imaginary1, c1.getImaginary(), 0.0);
        assertEquals(real2, c2.getReal(), 0.0);
        assertEquals(imaginary2, c2.getImaginary(), 0.0);
        assertEquals(0.0, result.getReal(), 0.0);
        assertEquals(0.0, result.getImaginary(), 0.0);

        // divide and store in result
        c1.divide(c2, result);
        // check correctness
        final var resultModulus = modulus1 / modulus2;
        var resultPhase = phase1 - phase2;
        final var resultReal = resultModulus * Math.cos(resultPhase);
        final var resultImaginary = resultModulus * Math.sin(resultPhase);
        resultPhase = Math.atan2(resultImaginary, resultReal);
        assertEquals(resultModulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, result.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // divide and return result
        result = c1.divideAndReturnNew(c2);
        // check correctness
        assertEquals(resultModulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, result.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // divide and store result on same instance
        c1.divide(c2);
        // check correctness
        assertEquals(resultModulus, c1.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, c1.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, c1.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, c1.getImaginary(), ABSOLUTE_ERROR);
    }

    @Test
    void testMultiplyByScalar() {
        final var randomizer = new UniformRandomizer();
        final var real = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var scalar = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);

        final var c = new Complex(real, imaginary);
        var result = new Complex();

        assertEquals(real, c.getReal(), 0.0);
        assertEquals(imaginary, c.getImaginary(), 0.0);
        assertEquals(0.0, result.getReal(), 0.0);
        assertEquals(0.0, result.getImaginary(), 0.0);

        // multiply by scalar and store in result
        c.multiplyByScalar(scalar, result);
        // check correctness
        assertEquals(scalar * real, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(scalar * imaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // multiply by scalar and return result
        result = c.multiplyByScalarAndReturnNew(scalar);
        // check correctness
        assertEquals(scalar * real, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(scalar * imaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // multiply by scalar and store result on same instance
        c.multiplyByScalar(scalar);
        // check correctness
        assertEquals(scalar * real, c.getReal(), ABSOLUTE_ERROR);
        assertEquals(scalar * imaginary, c.getImaginary(), ABSOLUTE_ERROR);
    }

    @Test
    void testPow() {
        final var randomizer = new UniformRandomizer();
        final var modulus = randomizer.nextDouble(MIN_MODULUS, MAX_MODULUS);
        final var phase = randomizer.nextDouble(MIN_PHASE, MAX_PHASE);
        final var exponent = randomizer.nextDouble(MIN_EXPONENT, MAX_EXPONENT);

        final var real = modulus * Math.cos(phase);
        final var imaginary = modulus * Math.sin(phase);

        final var c = new Complex(real, imaginary);
        var result = new Complex();

        assertEquals(real, c.getReal(), 0.0);
        assertEquals(imaginary, c.getImaginary(), 0.0);
        assertEquals(0.0, result.getReal(), 0.0);
        assertEquals(0.0, result.getImaginary(), 0.0);

        // compute power and store in result
        c.pow(exponent, result);
        // check correctness
        final var resultModulus = Math.pow(modulus, exponent);
        var resultPhase = exponent * phase;
        final var resultReal = resultModulus * Math.cos(resultPhase);
        final var resultImaginary = resultModulus * Math.sin(resultPhase);
        resultPhase = Math.atan2(resultImaginary, resultReal);
        assertEquals(resultModulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, result.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // multiply and return result
        result = c.powAndReturnNew(exponent);
        // check correctness
        assertEquals(resultModulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, result.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // multiply and store result on same instance
        c.pow(exponent);
        // check correctness
        assertEquals(resultModulus, c.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, c.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, c.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, c.getImaginary(), ABSOLUTE_ERROR);
    }

    @Test
    void testSqrt() {
        final var randomizer = new UniformRandomizer();
        final var modulus = randomizer.nextDouble(MIN_MODULUS, MAX_MODULUS);
        final var phase = randomizer.nextDouble(MIN_PHASE, MAX_PHASE);

        final var real = modulus * Math.cos(phase);
        final var imaginary = modulus * Math.sin(phase);

        final var c = new Complex(real, imaginary);
        var result = new Complex();

        assertEquals(real, c.getReal(), 0.0);
        assertEquals(imaginary, c.getImaginary(), 0.0);
        assertEquals(0.0, result.getReal(), 0.0);
        assertEquals(0.0, result.getImaginary(), 0.0);

        // compute power and store in result
        c.sqrt(result);
        // check correctness
        final var resultModulus = Math.sqrt(modulus);
        var resultPhase = 0.5 * phase;
        final var resultReal = resultModulus * Math.cos(resultPhase);
        final var resultImaginary = resultModulus * Math.sin(resultPhase);
        resultPhase = Math.atan2(resultImaginary, resultReal);
        assertEquals(resultModulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, result.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // multiply and return result
        result = c.sqrtAndReturnNew();
        // check correctness
        assertEquals(resultModulus, result.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, result.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, result.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, result.getImaginary(), ABSOLUTE_ERROR);

        // multiply and store result on same instance
        c.sqrt();
        // check correctness
        assertEquals(resultModulus, c.getModulus(), ABSOLUTE_ERROR);
        assertEquals(resultPhase, c.getPhase(), ABSOLUTE_ERROR);
        assertEquals(resultReal, c.getReal(), ABSOLUTE_ERROR);
        assertEquals(resultImaginary, c.getImaginary(), ABSOLUTE_ERROR);
    }

    @Test
    void testEqualsAndHashCode() {
        final var randomizer = new UniformRandomizer();
        final var real1 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var real2 = 1.0 + real1;
        final var imaginary1 = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary2 = 1.0 + imaginary1;

        final var c1 = new Complex(real1, imaginary1);
        final var c2 = new Complex(real1, imaginary1);
        final var c3 = new Complex(real2, imaginary2);

        //noinspection EqualsWithItself
        assertEquals(c1, c1);
        assertEquals(c1, c2);
        assertNotEquals(c1, c3);
        assertNotEquals(c1, new Object());

        assertEquals(c1.hashCode(), c2.hashCode());
    }

    @Test
    void testClone() {
        final var randomizer = new UniformRandomizer();
        final var real = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var c1 = new Complex(real, imaginary);
        final var c2 = new Complex(c1);

        assertEquals(c1, c2);
    }

    @Test
    void testCopyFrom() {
        final var randomizer = new UniformRandomizer();
        final var real = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var c1 = new Complex(real, imaginary);
        final var c2 = new Complex();

        assertEquals(real, c1.getReal(), 0.0);
        assertEquals(imaginary, c1.getImaginary(), 0.0);

        assertEquals(0.0, c2.getReal(), 0.0);
        assertEquals(0.0, c2.getImaginary(), 0.0);

        // copy c1 into c2
        c2.copyFrom(c1);

        // check correctness
        assertEquals(real, c2.getReal(), 0.0);
        assertEquals(imaginary, c2.getImaginary(), 0.0);
    }

    @Test
    void testSerializeDeserialize() throws IOException, ClassNotFoundException {
        final var randomizer = new UniformRandomizer();
        final var real = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var imaginary = randomizer.nextDouble(MIN_RANDOM_VALUE, MAX_RANDOM_VALUE);
        final var c1 = new Complex(real, imaginary);

        final var bytes = SerializationHelper.serialize(c1);
        final var c2 = SerializationHelper.deserialize(bytes);

        assertEquals(c1, c2);
        assertNotSame(c1, c2);
    }
}
