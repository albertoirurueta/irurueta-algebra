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

public class DecomposerHelper {

    private static final int MIN_VALUE = 1;
    private static final int MAX_VALUE = 100;
    private static final int MIN_SPD_VALUE = 1;
    private static final int MAX_SPD_VALUE = 4;

    private static final double ERROR = 1e-8;

    // Cholesky Decomposer
    public static Matrix getLeftLowerTriangulatorFactor(final int rows) throws WrongSizeException {

        // Symmetric positive definite matrices are square symmetric matrices
        // having all their eigenvalues positive or zero.
        // Cholesky decomposition is unique for symmetric positive definite
        // matrices allowing factorization of A into: A = L * L'
        // where L is a lower triangular matrix and the elements in the diagonal
        // of L are the square root of the eigenvalues, as can be seen from eigen
        // decomposition expression of a symmetric matrix: A = Q * D * Q', where
        // Q is an orthogonal matrix and D is a diagonal matrix containing
        // eigenvalues.
        final var randomizer = new UniformRandomizer();

        // Create random lower triangular matrix ensuring that elements in the
        // diagonal (eigenvalues), are positive to ensure positive definiteness
        final var l = new Matrix(rows, rows);

        for (var j = 0; j < rows; j++) {
            for (var i = 0; i < rows; i++) {
                if (i < j) {
                    l.setElementAt(i, j, 0.0);
                } else if (i == j) {
                    l.setElementAt(i, j, Math.abs(randomizer.nextDouble(MIN_SPD_VALUE, MAX_SPD_VALUE)));
                } else {
                    l.setElementAt(i, j, randomizer.nextDouble(MIN_SPD_VALUE, MAX_SPD_VALUE));
                }
            }
        }
        return l;
    }

    public static Matrix getSymmetricPositiveDefiniteMatrixInstance(final Matrix l) throws WrongSizeException {
        return l.multiplyAndReturnNew(l.transposeAndReturnNew());
    }

    public static Matrix getSingularMatrixInstance(final int rows, final int columns) throws WrongSizeException {
        final var randomizer = new UniformRandomizer();
        final var m = new Matrix(rows, columns);
        final var length = rows * columns;
        final var row1 = randomizer.nextInt(0, rows);
        int row2;
        do {
            row2 = randomizer.nextInt(0, rows);
        } while (row2 == row1);

        // Initialize matrix with 2 ld rows
        for (var i = 0; i < length; i++) {
            m.setElementAtIndex(i, randomizer.nextDouble(MIN_VALUE, MAX_VALUE), false);
        }

        for (var j = 0; j < m.getColumns(); j++) {
            m.setElementAt(row2, j, m.getElementAt(row1, j));
        }

        return m;
    }

    // LU, QR and Economy QR decomposers
    public static Matrix getSymmetricMatrix(final int rows) throws WrongSizeException {

        // generate random matrix of size rows x rows (the number of columns can
        // be anything indeed)
        final var m = Matrix.createWithUniformRandomValues(rows, rows, MIN_VALUE, MAX_VALUE);

        // create a symmetric matrix by multiplying it with its transpose
        m.multiply(m.transposeAndReturnNew());
        return m;
    }

    static Matrix getNonSingularMatrixInstance(final int rows, final int columns) throws WrongSizeException {
        final var randomizer = new UniformRandomizer();

        final var m = new Matrix(rows, columns);
        if (rows < 2 || columns < 2) {
            for (var i = 0; i < m.getRows(); i++) {
                for (var j = 0; j < m.getColumns(); j++) {
                    m.setElementAt(i, j, randomizer.nextDouble(MIN_VALUE, MAX_VALUE));
                }
            }
        } else {
            boolean ld;
            final var rowA = new double[m.getColumns()];
            final var rowB = new double[m.getColumns()];

            for (var i = 0; i < m.getRows(); i++) {
                ld = false;
                do {
                    for (var j = 0; j < m.getColumns(); j++) {
                        // Assign random values
                        m.setElementAt(i, j, randomizer.nextDouble(MIN_VALUE, MAX_VALUE));
                    }

                    if (i > 0) {
                        // Check that current row is not proportional to previous
                        // This is useful for matrices of size (2, 2)
                        ld = true;
                        for (var j2 = 0; j2 < m.getColumns(); j2++) {
                            if (Math.abs(m.getElementAt(i, 0) * m.getElementAt(i - 1, j2)
                                    - m.getElementAt(i - 1, 0) * m.getElementAt(i, j2)) > ERROR) {
                                ld = false;
                                break;
                            }
                        }
                    }
                    // Check that current row is not LD with any two previous
                    // ones
                    for (var i2 = 0; i2 < i - 1; i2++) {
                        for (var j2 = 0; j2 < m.getColumns(); j2++) {
                            rowA[j2] = m.getElementAt(i2 + 1, 0) * m.getElementAt(i2, j2)
                                    - m.getElementAt(i2, 0) * m.getElementAt(i2 + 1, j2);
                            rowB[j2] = m.getElementAt(i, 0) * m.getElementAt(i2, j2)
                                    - m.getElementAt(i2, 0) * m.getElementAt(i, j2);
                        }

                        // Check whether 2 rows are LD
                        ld = true;
                        for (var j2 = 0; j2 < m.getColumns(); j2++) {
                            if (Math.abs(rowA[j2] * rowB[m.getColumns() - 1]
                                    - rowB[j2] * rowA[m.getColumns() - 1]) > ERROR) {
                                ld = false;
                                break;
                            }
                        }
                    }
                } while (ld);
            }
        }
        return m;
    }


    static Matrix getOrthonormalMatrix(final int rows) throws WrongSizeException {

        final var m = Matrix.identity(rows, rows);
        final var value = 1.0 / Math.sqrt(rows);
        m.multiplyByScalar(value);

        // now scramble an amount of columns
        final var randomizer = new UniformRandomizer();
        final var column1 = new double[rows];
        final var column2 = new double[rows];

        for (var i = 0; i < rows; i++) {
            final var originColumn = randomizer.nextInt(0, rows);
            final var destinationColumn = randomizer.nextInt(0, rows);
            m.getSubmatrixAsArray(0, originColumn, rows - 1, originColumn, column1);
            m.getSubmatrixAsArray(0, destinationColumn, rows - 1, destinationColumn, column2);

            // now swap columns
            m.setSubmatrix(0, originColumn, rows - 1, originColumn, column2);
            m.setSubmatrix(0, destinationColumn, rows - 1, destinationColumn, column1);
        }

        return m;
    }
}
