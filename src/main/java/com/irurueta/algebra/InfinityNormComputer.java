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

/**
 * Class in charge of computing infinity norms of vectors and matrices.
 * Infinity norm is defined as the maximum row sum on a matrix or vector.
 * <p>
 * For the case of arrays, this library considers arrays as column matrices
 * with one column and array length rows, hence absolute value of first element
 * in array is returned as infinity norm.
 */
public class InfinityNormComputer extends NormComputer {

    /**
     * Constructor of this class.
     */
    public InfinityNormComputer() {
        super();
    }

    /**
     * Returns norm type being used by this class.
     *
     * @return Norm type being used by this class.
     */
    @Override
    public NormType getNormType() {
        return NormType.INFINITY_NORM;
    }

    /**
     * Computes norm of provided matrix.
     *
     * @param m matrix being used for norm computation.
     * @return norm of provided matrix.
     */
    @SuppressWarnings("DuplicatedCode")
    public static double norm(final Matrix m) {
        final var rows = m.getRows();
        final var columns = m.getColumns();
        double rowSum;
        var maxRowSum = 0.0;

        for (var i = 0; i < rows; i++) {
            rowSum = 0.0;
            for (var j = 0; j < columns; j++) {
                rowSum += Math.abs(m.getElementAt(i, j));
            }

            maxRowSum = Math.max(rowSum, maxRowSum);
        }

        return maxRowSum;
    }

    /**
     * Computes norm of provided array and stores the jacobian into provided
     * instance.
     *
     * @param array    array being used for norm computation.
     * @param jacobian instance where jacobian will be stored. Must be 1xN,
     *                 where N is length of array.
     * @return norm of provided vector.
     * @throws WrongSizeException if provided jacobian is not 1xN, where N is
     *                            length of array.
     */
    public static double norm(final double[] array, final Matrix jacobian) throws WrongSizeException {
        if (jacobian != null && (jacobian.getRows() != 1 || jacobian.getColumns() != array.length)) {
            throw new WrongSizeException();
        }

        final var norm = InfinityNormComputer.norm(array);

        if (jacobian != null) {
            jacobian.fromArray(array);
            jacobian.multiplyByScalar(1.0 / norm);
        }

        return norm;
    }

    /**
     * Computes norm of provided matrix.
     *
     * @param m Matrix being used for norm computation.
     * @return Norm of provided matrix.
     */
    @Override
    public double getNorm(final Matrix m) {
        return norm(m);
    }

    /**
     * Computes norm of provided array.
     *
     * @param array array being used for norm computation.
     * @return norm of provided vector.
     */
    public static double norm(final double[] array) {
        return Math.abs(array[0]);
    }

    /**
     * Computes norm of provided array.
     *
     * @param array Array being used for norm computation.
     * @return Norm of provided vector.
     */
    @Override
    public double getNorm(final double[] array) {
        return norm(array);
    }
}
