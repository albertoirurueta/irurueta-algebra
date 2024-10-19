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
 * Class in charge of computing norms of arrays and matrices.
 * Norms can be computed in different ways depending on the desired norm
 * measure. For that purpose subclass implementations of this class attempt to
 * work on different norm types.
 * By default, Frobenius norm is used for arrays and matrices, which consists
 * on computing the square root of summation of all the squared elements within
 * a given matrix or array.
 */
public abstract class NormComputer {
    /**
     * Constant defining default norm type to be used.
     */
    public static final NormType DEFAULT_NORM_TYPE = NormType.FROBENIUS_NORM;

    /**
     * Constructor of this class.
     */
    protected NormComputer() {
    }

    /**
     * Returns norm type being used by this class.
     *
     * @return Norm type being used by this class.
     */
    public abstract NormType getNormType();

    /**
     * Computes norm of provided matrix.
     *
     * @param m Matrix being used for norm computation.
     * @return Norm of provided matrix.
     */
    public abstract double getNorm(final Matrix m);

    /**
     * Computes norm of provided array.
     *
     * @param array Array being used for norm computation.
     * @return Norm of provided vector.
     */
    public abstract double getNorm(final double[] array);

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
    public double getNorm(final double[] array, final Matrix jacobian) throws WrongSizeException {
        if (jacobian != null && (jacobian.getRows() != 1 || jacobian.getColumns() != array.length)) {
            throw new WrongSizeException("jacobian must be 1xN, where N is length of array");
        }

        final var norm = getNorm(array);

        if (jacobian != null) {
            jacobian.fromArray(array);
            jacobian.multiplyByScalar(1.0 / norm);
        }

        return norm;
    }

    /**
     * Factory method. Returns a new instance of NormComputer prepared to
     * compute norms using provided NormType.
     *
     * @param normType Norm type to be used by returned instance.
     * @return New instance of NormComputer.
     */
    public static NormComputer create(final NormType normType) {
        return switch (normType) {
            case INFINITY_NORM -> new InfinityNormComputer();
            case ONE_NORM -> new OneNormComputer();
            default -> new FrobeniusNormComputer();
        };
    }

    /**
     * Factory method. Returns a new instance of NormComputer prepared to
     * compute norms using provided NormType.
     *
     * @return New instance of NormComputer.
     */
    public static NormComputer create() {
        return create(DEFAULT_NORM_TYPE);
    }
}
