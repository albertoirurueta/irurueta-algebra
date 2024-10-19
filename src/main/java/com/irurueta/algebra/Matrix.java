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

import com.irurueta.statistics.GaussianRandomizer;
import com.irurueta.statistics.UniformRandomizer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/**
 * Defines a matrix of numerical data.
 * Values of a matrix are stored inside an internal array of data.
 */
@SuppressWarnings("DuplicatedCode")
public class Matrix implements Serializable, Cloneable {
    /**
     * Constant defining the default order in which values are stored in a
     * matrix. This can be useful when copying data to and from an array.
     */
    public static final boolean DEFAULT_USE_COLUMN_ORDER = true;

    /**
     * Number of matrix rows.
     */
    private int rows;

    /**
     * Number of matrix columns.
     */
    private int columns;

    /**
     * Array containing data of matrix. Data is stored linearly in memory
     * using column order.
     */
    private double[] buffer;

    /**
     * Array used for indexing column start positions within buffer. This
     * is used for faster access to matrix elements.
     */
    private int[] columnIndex;

    /**
     * Constructor of this class.
     * By default, when instantiating a matrix its values are undefined.
     *
     * @param rows    Defines number of rows in matrix.
     * @param columns Defines number of columns in matrix.
     * @throws WrongSizeException Exception thrown if matrix must be
     *                            empty with provided size. In other words, matrices must have at least
     *                            one row and one column. Leaving number of rows or columns to zero will
     *                            raise this exception.
     */
    public Matrix(final int rows, final int columns) throws WrongSizeException {
        internalResize(rows, columns);
    }

    /**
     * Copy constructor.
     *
     * @param m matrix to copy from.
     */
    public Matrix(final Matrix m) {
        try {
            internalResize(m.getRows(), m.getColumns());
        } catch (final WrongSizeException ignore) {
            // never happens
        }

        System.arraycopy(m.getBuffer(), 0, buffer, 0, buffer.length);
    }

    /**
     * Returns number of rows in matrix.
     *
     * @return Number of rows in matrix.
     */
    public int getRows() {
        return rows;
    }

    /**
     * Returns number of columns in matrix.
     *
     * @return Number of columns in matrix.
     */
    public int getColumns() {
        return columns;
    }

    /**
     * Obtains element in matrix located at position (row and column).
     * Notice that row and column position are zero-indexed.
     *
     * @param row    Row to be used for element location.
     * @param column Column to be used for element location.
     * @return Value of element in matrix located at provided position.
     * @throws ArrayIndexOutOfBoundsException Exception raised if attempting
     *                                        to access a location that lies outside the boundaries of the internal
     *                                        array containing matrix data. Notice that internal data is stored in
     *                                        column order, hence, if row position exceeds the number of rows in the
     *                                        matrix, this exception might not be raised depending on column position,
     *                                        however, if the column value exceeds the number of columns, it will
     *                                        always raise the exception.
     */
    public double getElementAt(final int row, final int column) {
        return buffer[columnIndex[column] + row];
    }

    /**
     * Returns index within internal buffer corresponding to provided row
     * and column positions.
     *
     * @param row    row to be used for element location.
     * @param column column to be used for element location.
     * @return index within internal buffer.
     */
    public int getIndex(final int row, final int column) {
        return columnIndex[column] + row;
    }

    /**
     * Obtains element in matrix located at provided index value using the
     * order specified by DEFAULT_USE_COLUMN_ORDER.
     * Note: an index indicates linear position within matrix, in layman terms,
     * if using column order, then position (i, j), where i stands for row
     * index and j for column index, becomes index = j * rows + 1. On the other
     * hand, if using row order then index becomes index = i * columns + j.
     *
     * @param index Linear position.
     * @return Value of element contained at provided position.
     * @throws ArrayIndexOutOfBoundsException Exception raised if index lays
     *                                        outside valid values, which range from zero (inclusive) to
     *                                        rows * columns (exclusive)
     */
    public double getElementAtIndex(final int index) {
        return getElementAtIndex(index, DEFAULT_USE_COLUMN_ORDER);
    }

    /**
     * Obtain element in matrix located at provided index value using provided
     * order (either column or row order).
     *
     * @param index         Linear position.
     * @param isColumnOrder if true indicates that values are retrieved
     *                      assuming that are stored in column order.
     * @return Value of element contained at provided position.
     * @throws ArrayIndexOutOfBoundsException Exception raised if index lays
     *                                        outside valid values, which range from zero (inclusive) to
     *                                        rows * columns (exclusive)
     */
    public double getElementAtIndex(final int index, final boolean isColumnOrder) {

        if (isColumnOrder) {
            return buffer[index];
        } else {
            final var row = index / columns;
            final var column = index % columns;
            return buffer[columnIndex[column] + row];
        }
    }

    /**
     * Sets element in matrix located at provided position (row and column).
     *
     * @param row    Row to be used for element location to be set.
     * @param column Column to be used for element location to be set.
     * @param value  Value to be set at provided position.
     * @throws ArrayIndexOutOfBoundsException Exception raised if attempting
     *                                        to access a location that lies outside the boundaries of the internal
     *                                        array containing matrix data. Notice that internal data is stored in
     *                                        column order, hence, if row position exceeds the number of rows in the
     *                                        matrix, this exception might not be raised depending on column position,
     *                                        however, if the column value exceeds the number of columns, it will
     *                                        always raise the exception.
     */
    public void setElementAt(final int row, final int column, final double value) {
        buffer[columnIndex[column] + row] = value;
    }

    /**
     * Sets element in matrix located at provided index using the order
     * specified by DEFAULT_USE_COLUMN_ORDER.
     * Note: an index indicates linear position within matrix, in layman terms,
     * if using column order, then position (i, j), where i stands for row
     * index and j for column index, becomes index = j * rows + 1. On the other
     * hand, if using row order then index becomes index = i * columns + j.
     *
     * @param index Linear position.
     * @param value Value of element contained at provided position.
     * @throws ArrayIndexOutOfBoundsException Exception raised if index lays
     *                                        outside valid values, which range from zero (inclusive) to
     *                                        rows * columns (exclusive)
     */
    public void setElementAtIndex(final int index, final double value) {
        setElementAtIndex(index, value, DEFAULT_USE_COLUMN_ORDER);
    }

    /**
     * Sets element in matrix located at provided index using provided order
     * (either column or row order).
     *
     * @param index         Linear position.
     * @param value         Value of element to be set at provided position.
     * @param isColumnOrder if true indicates that values are retrieved
     *                      assuming that are stored in column order.
     * @throws ArrayIndexOutOfBoundsException Exception raised if index lays
     *                                        outside valid values, which range from zero (inclusive) to
     *                                        rows * columns (exclusive).
     */
    public void setElementAtIndex(final int index, final double value, final boolean isColumnOrder) {
        if (isColumnOrder) {
            buffer[index] = value;
        } else {
            final var row = index / columns;
            final var column = index % columns;
            buffer[columnIndex[column] + row] = value;
        }
    }

    /**
     * Returns a new matrix instance containing the same data as this instance.
     *
     * @return A copy of this matrix instance.
     * @throws CloneNotSupportedException if clone fails.
     */
    @Override
    public Matrix clone() throws CloneNotSupportedException {
        var out = (Matrix) super.clone();
        out.copyFrom(this);
        return out;
    }

    /**
     * Copies this matrix data into provided matrix. Provided output matrix will
     * be resized if needed.
     *
     * @param output Destination matrix where data will be copied to.
     * @throws NullPointerException Exception raised if provided output matrix
     *                              is null.
     */
    public void copyTo(final Matrix output) {
        // reset output size if not equal, otherwise reuse buffer and column
        // index
        if (output.getRows() != rows || output.getColumns() != columns) {
            // resets size and column index
            try {
                output.resize(rows, columns);
            } catch (final WrongSizeException ignore) {
                // never happens
            }
        }
        // copies content
        System.arraycopy(buffer, 0, output.buffer, 0, buffer.length);
    }

    /**
     * Copies the contents of provided matrix into this instance. This instance
     * will be resized if needed.
     *
     * @param input Input matrix where data will be copied from.
     * @throws NullPointerException Exception raised if provided input matrix is
     *                              null.
     */
    public void copyFrom(final Matrix input) {
        // reset size if not equal, otherwise reuse buffer and column index
        if (input.getRows() != rows || input.getColumns() != columns) {
            // resets size and column index
            try {
                resize(input.getRows(), input.getColumns());
            } catch (final WrongSizeException ignore) {
                // never happens
            }
        }
        // copies content
        System.arraycopy(input.buffer, 0, buffer, 0, buffer.length);
    }

    /**
     * Adds another matrix to this matrix instance and stores the result in
     * provided result matrix. If provided result matrix doesn't have proper
     * size, it will be resized.
     *
     * @param other  Matrix to be added to current instance.
     * @param result Matrix where result of summation is stored.
     * @throws WrongSizeException   Exception thrown if provided matrix to be
     *                              added (i.e. other) does not have the same size as this matrix.
     * @throws NullPointerException Exception raised if provided matrices are
     *                              null.
     */
    public void add(final Matrix other, final Matrix result) throws WrongSizeException {
        if (other.getRows() != rows || other.getColumns() != columns) {
            throw new WrongSizeException();
        }

        // resize result if needed
        if (result.getRows() != rows || result.getColumns() != columns) {
            result.resize(rows, columns);
        }

        internalAdd(other, result);
    }

    /**
     * Adds provided matrix to this instance and returns the result as a new
     * matrix instance.
     *
     * @param other Matrix to be added.
     * @return Returns a new matrix containing the sum of this matrix with
     * provided matrix.
     * @throws WrongSizeException   Exception raised if provided matrix does
     *                              not have the same size as this matrix.
     * @throws NullPointerException Exception raised if provided matrix is null.
     */
    public Matrix addAndReturnNew(final Matrix other) throws WrongSizeException {
        if (other.getRows() != rows || other.getColumns() != columns) {
            throw new WrongSizeException();
        }

        final var out = new Matrix(rows, columns);
        internalAdd(other, out);
        return out;
    }

    /**
     * Adds provided matrix to this instance.
     *
     * @param other Matrix to be added.
     * @throws WrongSizeException   Exception raised if provided matrix does
     *                              not have the same size as this matrix.
     * @throws NullPointerException Exception raised if provided matrix is null.
     */
    public void add(final Matrix other) throws WrongSizeException {
        if (other.getRows() != rows || other.getColumns() != columns) {
            throw new WrongSizeException();
        }

        internalAdd(other, this);
    }

    /**
     * Subtracts another matrix from this matrix instance and stores the result
     * in provided result matrix. If provided result matrix doesn't have proper
     * size, it will be resized.
     *
     * @param other  Matrix to be added to current instance.
     * @param result Matrix where result of subtraction is stored.
     * @throws WrongSizeException   Exception thrown if provided matrix to be
     *                              subtracted (i.e. other) does not have the same size as this matrix.
     * @throws NullPointerException Exception raised if provided matrices are
     *                              null.
     */
    public void subtract(final Matrix other, final Matrix result) throws WrongSizeException {
        if (other.getRows() != rows || other.getColumns() != columns) {
            throw new WrongSizeException();
        }

        // resize result if needed
        if (result.getRows() != rows || result.getColumns() != columns) {
            result.resize(rows, columns);
        }

        internalSubtract(other, result);
    }

    /**
     * Subtracts provided matrix from this instance and returns the result as a
     * new matrix instance.
     *
     * @param other Matrix to be subtracted from.
     * @return Returns a new matrix containing the subtraction of provided
     * matrix from this matrix.
     * @throws WrongSizeException   Exception raised if provided matrix does
     *                              not have the same size as this matrix.
     * @throws NullPointerException Exception raised if provided matrix is null.
     */
    public Matrix subtractAndReturnNew(final Matrix other) throws WrongSizeException {
        if (other.getRows() != rows || other.getColumns() != columns) {
            throw new WrongSizeException();
        }

        final var out = new Matrix(rows, columns);
        internalSubtract(other, out);
        return out;
    }

    /**
     * Subtracts provided matrix from this instance.
     *
     * @param other Matrix to be subtracted from.
     * @throws WrongSizeException   Exception raised if provided matrix does
     *                              not have the same size as this matrix.
     * @throws NullPointerException Exception raised if provided matrix is null.
     */
    public void subtract(final Matrix other) throws WrongSizeException {
        if (other.getRows() != rows || other.getColumns() != columns) {
            throw new WrongSizeException();
        }

        internalSubtract(other, this);
    }

    /**
     * Multiplies another matrix to this matrix instance and stores the result
     * in provided result matrix. If provided result matrix doesn't have proper
     * size, it will be resized.
     *
     * @param other  Matrix to be multiplied to current instance.
     * @param result Matrix where result of product is stored.
     * @throws WrongSizeException   Exception thrown when current and provided
     *                              matrix (i.e. other) has incompatible size for product computation.
     * @throws NullPointerException Exception raised if provided matrices are
     *                              null.
     */
    public void multiply(final Matrix other, final Matrix result) throws WrongSizeException {
        if (columns != other.rows) {
            throw new WrongSizeException();
        }

        // resize result if needed
        if (result.rows != rows || result.columns != other.columns) {
            result.resize(rows, other.columns);
        }

        internalMultiply(other, result);
    }

    /**
     * Multiplies this matrix with provided matrix and returns the result as
     * a new instance.
     * If this matrix m1 has size m x n and provided matrix m2 has size p x q,
     * then n must be equal to p so that product m1 * m2 can be correctly
     * computed obtaining a matrix of size m x q, otherwise an
     * IllegalArgumentException will be raised.
     *
     * @param other Right operand of matrix product
     * @return Matrix containing result of multiplication
     * @throws WrongSizeException   Exception thrown when current and
     *                              provided matrices have incompatible sizes for product computation.
     * @throws NullPointerException Exception thrown if provided matrix is null
     */
    public Matrix multiplyAndReturnNew(final Matrix other) throws WrongSizeException {

        if (columns != other.rows) {
            throw new WrongSizeException();
        }

        final var out = new Matrix(rows, other.columns);
        internalMultiply(other, out);
        return out;
    }

    /**
     * Multiplies this matrix with provided matrix.
     * If this matrix m1 has size m x n and provided matrix m2 has size p x q,
     * then n must be equal to p so that product m1 * m2 can be correctly
     * computed resizing this matrix to a new one having size m x q, otherwise
     * an IllegalArgumentException will be raised.
     *
     * @param other Right operand of matrix product
     * @throws WrongSizeException   Exception thrown when current and
     *                              provided matrices have incompatible sizes for product computation.
     * @throws NullPointerException Exception thrown if provided matrix is null
     */
    public void multiply(final Matrix other) throws WrongSizeException {

        if (columns != other.rows) {
            throw new WrongSizeException();
        }

        // instantiate new buffer and column index
        final var resultBuffer = new double[rows * other.columns];
        final var resultColumnIndex = new int[other.columns];
        var counter = 0;
        for (int i = 0; i < other.columns; i++) {
            resultColumnIndex[i] = counter;
            counter += rows;
        }

        internalMultiply(other, resultBuffer, resultColumnIndex);
        // update matrix data
        columns = other.columns;
        columnIndex = resultColumnIndex;
        buffer = resultBuffer;
    }

    /**
     * Computes the Kronecker product with provided matrix and stores the
     * result in provided result matrix. If provided result matrix doesn't
     * have proper size, it will be resized.
     *
     * @param other  other matrix to be Kronecker multiplied to current matrix.
     * @param result matrix where result will be stored.
     */
    public void multiplyKronecker(final Matrix other, final Matrix result) {
        // resize result if needed
        final var resultRows = rows * other.rows;
        final var resultCols = columns * other.columns;
        if (result.rows != resultRows || result.columns != resultCols) {
            try {
                result.resize(resultRows, resultCols);
            } catch (final WrongSizeException ignore) {
                // never thrown
            }
        }

        internalMultiplyKronecker(other, result);
    }

    /**
     * Computes the Kronecker product with provided matrix and returns the
     * result as a new instance.
     * If this matrix m1 has size mxn and provided matrix m2 has size pxq,
     * the resulting matrix will be m*pxn*q.
     *
     * @param other other matrix to be Kronecker multiplied to current matrix.
     * @return matrix containing result of Kronecker multiplication.
     */
    public Matrix multiplyKroneckerAndReturnNew(final Matrix other) {
        Matrix out = null;
        try {
            out = new Matrix(rows * other.rows, columns * other.columns);
            internalMultiplyKronecker(other, out);
        } catch (final WrongSizeException ignore) {
            // never thrown
        }
        return out;
    }

    /**
     * Computes the Kronecker product of this matrix with provided matrix and
     * updates this matrix with the result of the multiplication.
     * If this matrix m1 has size mxn and provided matrix m2 has size pxq,
     * the resulting matrix will be resized to m*pxn*q.
     *
     * @param other other matrix to be Kronecker multiplied to current matrix.
     */
    public void multiplyKronecker(final Matrix other) {

        // instantiate new buffer and column index
        final var resultRows = rows * other.rows;
        final var resultCols = columns * other.columns;
        final var resultBuffer = new double[resultRows * resultCols];
        final var resultColumnIndex = new int[resultCols];
        var counter = 0;
        for (int i = 0; i < resultCols; i++) {
            resultColumnIndex[i] = counter;
            counter += resultRows;
        }

        internalMultiplyKronecker(other, resultBuffer, resultColumnIndex);
        // update matrix data
        rows = resultRows;
        columns = resultCols;
        columnIndex = resultColumnIndex;
        buffer = resultBuffer;
    }

    /**
     * Computes product by scalar of this instance multiplying all its elements
     * by provided scalar value and storing the results in provided result
     * matrix. If provided result matrix doesn't have proper size, it will be
     * automatically resized.
     *
     * @param scalar Scalar amount that current matrix will be multiplied by.
     * @param result Matrix where result of operation is stored.
     */
    private void multiplyByScalar(final double scalar, final Matrix result) {
        final var length = rows * columns;
        for (int i = 0; i < length; i++) {
            result.buffer[i] = scalar * buffer[i];
        }
    }

    /**
     * Computes product by scalar of this instance multiplying all its elements
     * by provided scalar value and returning the result as a new instance.
     *
     * @param scalar Scalar amount that current matrix will be multiplied by.
     * @return Returns a new matrix instance that contains result of product by
     * scalar.
     */
    public Matrix multiplyByScalarAndReturnNew(final double scalar) {
        Matrix out = null;
        try {
            out = new Matrix(rows, columns);
            multiplyByScalar(scalar, out);
        } catch (final WrongSizeException ignore) {
            // never happens
        }
        return out;
    }

    /**
     * Computes product by scalar of this instance multiplying all its elements
     * by provided scalar value and returning the result as a new instance.
     *
     * @param scalar Scalar amount that current matrix will be multiplied by.
     */
    public void multiplyByScalar(final double scalar) {
        multiplyByScalar(scalar, this);
    }

    /**
     * Checks if provided object is a Matrix instance having exactly the same
     * contents as this matrix instance.
     *
     * @param obj Object to be compared
     * @return Returns true if both objects are considered to be equal.
     */
    @Override
    public boolean equals(final Object obj) {
        if (obj == null) {
            return false;
        }
        if (obj == this) {
            return true;
        }
        if (!(obj instanceof Matrix other)) {
            return false;
        }

        return equals(other);
    }

    /**
     * Computes and returns hash code for this instance. Hash codes are almost
     * unique values that are useful for fast classification and storage of
     * objects in collections.
     *
     * @return Hash code
     */
    @Override
    public int hashCode() {
        return Objects.hash(this.rows, this.columns, Arrays.hashCode(this.buffer), Arrays.hashCode(this.columnIndex));
    }

    /**
     * Checks if provided matrix has exactly the same contents as this matrix
     * instance.
     *
     * @param other Matrix to be compared.
     * @return Returns true if both objects are considered to be equal (same
     * content and size)
     */
    public boolean equals(final Matrix other) {
        return equals(other, 0.0);
    }

    /**
     * Checks if provided matrix has contents similar to this matrix by checking
     * that all values have a maximum difference equal to provided threshold and
     * same size.
     *
     * @param other     Matrix to be compared
     * @param threshold Maximum difference allowed between values on same
     *                  position to determine that matrices are equal
     * @return True if matrices are considered to be equal (almost equal content
     * and same size)
     */
    public boolean equals(final Matrix other, final double threshold) {
        if (other == null) {
            return false;
        }
        if (other.getRows() != rows) {
            return false;
        }
        if (other.getColumns() != columns) {
            return false;
        }

        // check contents
        final var length = rows * columns;
        for (int i = 0; i < length; i++) {
            if (Math.abs(buffer[i] - other.buffer[i]) > threshold) {
                return false;
            }
        }
        return true;
    }

    /**
     * Computes element by element product (i.e. Hadamard product) between
     * current and provided (i.e. other) matrix, and stores the result in
     * provided result matrix.
     *
     * @param other  Instance that will be used for element by element product
     *               with current instance.
     * @param result Matrix where result of operation is stored.
     * @throws WrongSizeException   Exception raised if attempting to perform
     *                              element by element product on matrices of different size.
     * @throws NullPointerException Exception raised if provided matrix is null
     */
    public void elementByElementProduct(final Matrix other, final Matrix result) throws WrongSizeException {
        if (other.getRows() != rows || other.getColumns() != columns) {
            throw new WrongSizeException();
        }

        // resize result if needed
        if (result.getRows() != rows || result.getColumns() != columns) {
            result.resize(rows, columns);
        }

        internalElementByElementProduct(other, result);
    }

    /**
     * Computes element by element product (i.e. Hadamard product) between
     * current and provided instances and returns the result as a new instance.
     * Example:
     * Having matrices
     * [1, 2, 3]		 [10, 11, 12]
     * m1 =	[4, 5, 6]	m2 = [13, 14, 15]
     * [7, 8, 9]		 [16, 17, 18]
     * Then their element by element product will be
     * [10,  22,  36 ]
     * m3 =    [52,  70,  90 ]
     * [112, 136, 162]
     * <p>
     * Note: Attempting to perform element product on matrices of different size
     * will raise an IllegalArgumentException
     *
     * @param other Instance that will be used for element by element product
     *              with current instance.
     * @return A new Matrix containing element by element product result
     * @throws WrongSizeException   Exception raised if attempting to
     *                              perform element by element product on matrices of different size.
     * @throws NullPointerException Exception raised if provided matrix is null
     */
    public Matrix elementByElementProductAndReturnNew(final Matrix other) throws WrongSizeException {
        if (other.getRows() != rows || other.getColumns() != columns) {
            throw new WrongSizeException();
        }

        final var out = new Matrix(rows, columns);
        internalElementByElementProduct(other, out);
        return out;
    }

    /**
     * Computes element by element product (i.e. Hadamard product) between
     * current and provided instances.
     * Example:
     * Having matrices
     * [1, 2, 3]		 [10, 11, 12]
     * m1 =	[4, 5, 6]	m2 = [13, 14, 15]
     * [7, 8, 9]		 [16, 17, 18]
     * Then their element by element product will be
     * [10,  22,  36 ]
     * m3 =    [52,  70,  90 ]
     * [112, 136, 162]
     * <p>
     * Note: Attempting to perform element product on matrices of different size
     * will raise an IllegalArgumentException
     *
     * @param other Instance that will be used for element by element product
     *              with current instance.
     * @throws WrongSizeException   Exception raised if attempting to
     *                              perform element by element product on matrices of different size.
     * @throws NullPointerException Exception raised if provided matrix is null
     */
    public void elementByElementProduct(final Matrix other) throws WrongSizeException {
        if (other.getRows() != rows || other.getColumns() != columns) {
            throw new WrongSizeException();
        }

        internalElementByElementProduct(other, this);
    }

    /**
     * Transposes current matrix and stores result in provided matrix. If
     * provided matrix doesn't have proper size, it will be resized.
     *
     * @param result Instance where transposed matrix is stored.
     */
    public void transpose(final Matrix result) {
        // resize result if needed
        if (result.getRows() != columns || result.getColumns() != rows) {
            try {
                result.resize(columns, rows);
            } catch (final WrongSizeException ignore) {
                // never happens
            }
        }
        internalTranspose(result);
    }

    /**
     * Transposes current matrix and returns result as a new instance.
     * Transposition of a matrix is done by exchanging rows and columns, in
     * Layman terms, given a matrix m1 with elements located at m1(i,j), where i
     * is the row index and j is the column index, then it follows that its
     * transposed matrix m2 has the following property m2(i,j) = m1(j, i).
     *
     * @return A new Matrix instance containing transposed matrix.
     */
    public Matrix transposeAndReturnNew() {
        Matrix out = null;
        try {
            out = new Matrix(columns, rows);
            internalTranspose(out);
        } catch (final WrongSizeException ignore) {
            // never happens
        }
        return out;
    }

    /**
     * Transposes current matrix.
     * Transposition of a matrix is done by exchanging rows and columns, in
     * Layman terms, given a matrix m1 with elements located at m1(i,j), where i
     * is the row index and j is the column index, then it follows that its
     * transposed matrix m2 has the following property m2(i,j) = m1(j, i).
     */
    public void transpose() {

        final var newBuffer = new double[rows * columns];
        final var newColumnIndex = new int[rows];
        var counter = 0;
        for (var i = 0; i < rows; i++) {
            newColumnIndex[i] = counter;
            counter += columns;
        }
        internalTranspose(newBuffer, newColumnIndex);

        // update matrix data

        // swap rows and columns
        final var tmp = rows;
        rows = columns;
        columns = tmp;

        columnIndex = newColumnIndex;
        buffer = newBuffer;
    }

    /**
     * Sets the contents of this matrix to provided value in all of its elements
     *
     * @param initValue Value to be set on all the elements of this matrix.
     */
    public void initialize(final double initValue) {
        // initialize buffer array to provided value
        Arrays.fill(buffer, initValue);
    }

    /**
     * Resizes current instance by removing its contents and resizing it to
     * provided size.
     *
     * @param rows    Number of rows to be set
     * @param columns Number of columns to be set
     * @throws WrongSizeException Exception raised if either rows or
     *                            columns is zero.
     */
    public void resize(final int rows, final int columns) throws WrongSizeException {
        internalResize(rows, columns);
    }

    /**
     * Resets current instance by removing its contents, resizing it to provided
     * size and setting all its elements to provided value.
     *
     * @param rows      Number of rows to be set
     * @param columns   Number of columns to be set
     * @param initValue Value to be set in all of its elements
     * @throws WrongSizeException Exception raised if either rows or
     *                            columns is zero.
     */
    public void reset(final int rows, final int columns, final double initValue) throws WrongSizeException {
        internalResize(rows, columns);
        initialize(initValue);
    }

    /**
     * Returns the contents of the matrix as an array of values using
     * DEFAULT_USE_COLUMN_ORDER to pick elements.
     *
     * @return Contents of matrix as an array
     */
    public double[] toArray() {
        return toArray(DEFAULT_USE_COLUMN_ORDER);
    }

    /**
     * Returns the contents of the matrix as an array of values using provided
     * order to pick elements.
     *
     * @param isColumnOrder If true, picks elements from matrix using column
     *                      order, otherwise row order is used.
     * @return Contents of matrix as an array,
     */
    public double[] toArray(final boolean isColumnOrder) {
        final var length = rows * columns;

        if (isColumnOrder) {
            return Arrays.copyOf(buffer, length);
        } else {
            final var out = new double[length];
            double value;
            var counter = 0;

            for (var j = 0; j < rows; j++) {
                for (var i = 0; i < columns; i++) {
                    value = buffer[columnIndex[i] + j];
                    out[counter] = value;
                    counter++;
                }
            }

            return out;
        }
    }

    /**
     * Copies the contents of the matrix to an array of values using column
     * order.
     *
     * @param result array where values will be copied to.
     * @throws WrongSizeException if provided result array does not have the
     *                            same number of elements as the matrix (i.e. rows x columns).
     */
    public void toArray(final double[] result) throws WrongSizeException {
        toArray(result, DEFAULT_USE_COLUMN_ORDER);
    }

    /**
     * Copies the contents of the matrix to an array of values using provided
     * order to pick elements.
     *
     * @param result        array where values will be copied to.
     * @param isColumnOrder if true, picks elements from matrix using column
     *                      order, otherwise row order is used.
     * @throws WrongSizeException if provided result array does not have the
     *                            same number of elements as the matrix (i.e. rows x columns).
     */
    public void toArray(final double[] result, final boolean isColumnOrder) throws WrongSizeException {
        if (result.length != buffer.length) {
            throw new WrongSizeException("result array must be equal to rows x columns");
        }

        if (isColumnOrder) {
            System.arraycopy(buffer, 0, result, 0, buffer.length);
        } else {
            double value;
            var counter = 0;

            for (var j = 0; j < rows; j++) {
                for (var i = 0; i < columns; i++) {
                    value = buffer[columnIndex[i] + j];
                    result[counter] = value;
                    counter++;
                }
            }
        }
    }

    /**
     * Returns current matrix internal buffer of data.
     *
     * @return Internal buffer of data.
     */
    public double[] getBuffer() {
        return buffer;
    }

    /**
     * Obtains a sub-matrix of current matrix instance. Sub-matrix is obtained by
     * copying all elements contained within provided coordinates (both top-left
     * and bottom-right points are included within sub-matrix).
     *
     * @param topLeftRow        Top-left row index where sub-matrix starts.
     * @param topLeftColumn     Top-left column index where sub-matrix starts.
     * @param bottomRightRow    Bottom-right row index where sub-matrix ends.
     * @param bottomRightColumn Bottom-right column index where sub-matrix ends.
     * @param result            Instance where sub-matrix data is stored.
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located below or at right side of bottom-right corner.
     * @throws NullPointerException     If provided result matrix is null.
     */
    public void getSubmatrix(final int topLeftRow, final int topLeftColumn, final int bottomRightRow,
                             final int bottomRightColumn, final Matrix result) {
        if (topLeftRow < 0 || topLeftRow >= rows || topLeftColumn < 0 || topLeftColumn >= columns
                || bottomRightRow < 0 || bottomRightRow >= rows || bottomRightColumn < 0 || bottomRightColumn >= columns
                || topLeftRow > bottomRightRow || topLeftColumn > bottomRightColumn) {
            throw new IllegalArgumentException();
        }

        final var subRows = bottomRightRow - topLeftRow + 1;
        final var subCols = bottomRightColumn - topLeftColumn + 1;
        if (result.getRows() != subRows || result.getColumns() != subCols) {
            // resize result
            try {
                result.resize(subRows, subCols);
            } catch (final WrongSizeException ignore) {
                // never happens
            }
        }
        internalGetSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, result);
    }

    /**
     * Obtains a sub-matrix of current matrix instance. Sub-matrix is obtained by
     * copying all elements contained within provided coordinates (both top-left
     * and bottom-right points are included within sub-matrix).
     *
     * @param topLeftRow        Top-left row index where sub-matrix starts.
     * @param topLeftColumn     Top-left column index where sub-matrix starts.
     * @param bottomRightRow    Bottom-right row index where sub-matrix ends.
     * @param bottomRightColumn Bottom-right column index where sub-matrix ends.
     * @return A new instance containing selected sub-matrix.
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located belo or at right side of bottom-right corner.
     */
    public Matrix getSubmatrix(final int topLeftRow, final int topLeftColumn,
                               final int bottomRightRow, final int bottomRightColumn) {
        if (topLeftRow < 0 || topLeftRow >= rows || topLeftColumn < 0 || topLeftColumn >= columns
                || bottomRightRow < 0 || bottomRightRow >= rows || bottomRightColumn < 0 || bottomRightColumn >= columns
                || topLeftRow > bottomRightRow || topLeftColumn > bottomRightColumn) {
            throw new IllegalArgumentException();
        }

        Matrix out;
        try {
            out = new Matrix(bottomRightRow - topLeftRow + 1, bottomRightColumn - topLeftColumn + 1);
        } catch (final WrongSizeException e) {
            throw new IllegalArgumentException(e);
        }
        internalGetSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, out);
        return out;
    }

    /**
     * Retrieves a sub-matrix of current matrix instance as an array of values
     * using column order and storing the result in provided array.
     * Sub-matrix is obtained by copying all elements contained within provided
     * coordinates (both top-left and bottom-right points are included within
     * sub-matrix).
     *
     * @param topLeftRow        Top-left row index where sub-matrix starts
     * @param topLeftColumn     Top-left column index where sub-matrix starts
     * @param bottomRightRow    Bottom-right row index where sub-matrix ends
     * @param bottomRightColumn Bottom-right column index where sub-matrix ends.
     * @param array             Array where sub-matrix data is stored.
     * @throws IllegalArgumentException Thrown if provided coordinates lie
     *                                  outside of matrix boundaries or if top-left corner is at the bottom or
     *                                  right side of bottom-right corner
     * @throws WrongSizeException       thrown if length of provided array does not
     *                                  match the number of elements to be extracted from this matrix
     */
    public void getSubmatrixAsArray(final int topLeftRow, final int topLeftColumn,
                                    final int bottomRightRow, final int bottomRightColumn,
                                    final double[] array) throws WrongSizeException {
        if (topLeftRow < 0 || topLeftRow >= rows || topLeftColumn < 0 || topLeftColumn >= columns || bottomRightRow < 0
                || bottomRightRow >= rows || bottomRightColumn < 0 || bottomRightColumn >= columns
                || topLeftRow > bottomRightRow || topLeftColumn > bottomRightColumn) {
            throw new IllegalArgumentException();
        }

        final var length = (bottomRightRow - topLeftRow + 1) * (bottomRightColumn - topLeftColumn + 1);
        if (array.length != length) {
            throw new WrongSizeException();
        }

        getSubmatrixAsArray(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, DEFAULT_USE_COLUMN_ORDER,
                array);
    }

    /**
     * Retrieves a sub-matrix of current matrix instance as an array of values
     * using provided column order and storing the result in provided array.
     * Sub-matrix is obtained by copying all elements contained within provided
     * coordinates (both top-left and bottom-right points are included within
     * sub-matrix).
     *
     * @param topLeftRow        Top-left row index where sub-matrix starts
     * @param topLeftColumn     Top-left column index where sub-matrix starts
     * @param bottomRightRow    Bottom-right row index where sub-matrix ends
     * @param bottomRightColumn Bottom-right column index where sub-matrix ends.
     * @param isColumnOrder     If true, picks elements from matrix using column
     *                          order, otherwise row order is used.
     * @param array             Array where sub-matrix data is stored.
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located below or at right side of bottom-right corner.
     * @throws WrongSizeException       If provided array doesn't have proper length,
     *                                  which must be equal to the amount of elements in desired sub-matrix.
     */
    public void getSubmatrixAsArray(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn,
            final boolean isColumnOrder, final double[] array) throws WrongSizeException {
        if (topLeftRow < 0 || topLeftRow >= rows || topLeftColumn < 0 || topLeftColumn >= columns || bottomRightRow < 0
                || bottomRightRow >= rows || bottomRightColumn < 0 || bottomRightColumn >= columns
                || topLeftRow > bottomRightRow || topLeftColumn > bottomRightColumn) {
            throw new IllegalArgumentException();
        }

        final var length = (bottomRightRow - topLeftRow + 1) * (bottomRightColumn - topLeftColumn + 1);
        if (array.length != length) {
            throw new WrongSizeException();
        }

        internalGetSubmatrixAsArray(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, isColumnOrder, array);
    }

    /**
     * Obtains sub-matrix of current matrix instance as an array of values using
     * DEFAULT_USE_COLUMN_ORDER. Array is obtained by copying all elements
     * contained within provided coordinates (both top-left
     * and bottom-right points are included within sub-matrix).
     *
     * @param topLeftRow        Top-left row index where sub-matrix starts
     * @param topLeftColumn     Top-left column index where sub-matrix starts
     * @param bottomRightRow    Bottom-right row index where sub-matrix ends
     * @param bottomRightColumn Bottom-right column index where sub-matrix ends.
     * @return An array containing sub-matrix elements
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located belo or at right side of bottom-right corner.
     */
    public double[] getSubmatrixAsArray(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn) {
        return getSubmatrixAsArray(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn,
                DEFAULT_USE_COLUMN_ORDER);
    }

    /**
     * Obtains sub-matrix of current matrix instance as an array of values using
     * provided order. Array is obtained by copying all elements
     * contained within provided coordinates (both top-left
     * and bottom-right points are included within sub-matrix).
     *
     * @param topLeftRow        Top-left row index where sub-matrix starts
     * @param topLeftColumn     Top-left column index where sub-matrix starts
     * @param bottomRightRow    Bottom-right row index where sub-matrix ends
     * @param bottomRightColumn Bottom-right column index where sub-matrix ends.
     * @param isColumnOrder     If true, picks elements from matrix using column
     *                          order, otherwise row order is used.
     * @return An array containing sub-matrix elements
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located below or at right side of bottom-right corner.
     */
    public double[] getSubmatrixAsArray(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn,
            final boolean isColumnOrder) {
        if (topLeftRow < 0 || topLeftRow >= rows || topLeftColumn < 0 || topLeftColumn >= columns || bottomRightRow < 0
                || bottomRightRow >= rows || bottomRightColumn < 0 || bottomRightColumn >= columns
                || topLeftRow > bottomRightRow || topLeftColumn > bottomRightColumn) {
            throw new IllegalArgumentException();
        }

        final var length = (bottomRightRow - topLeftRow + 1) * (bottomRightColumn - topLeftColumn + 1);

        final var out = new double[length];
        internalGetSubmatrixAsArray(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, isColumnOrder, out);
        return out;
    }

    /**
     * Copies elements from provided sub-matrix into this matrix at provided
     * location.
     *
     * @param topLeftRow        Top-left row index where sub-matrix copy starts
     * @param topLeftColumn     Top-left column index where sub-matrix copy starts
     * @param bottomRightRow    Bottom-right row index where sub-matrix copy ends
     * @param bottomRightColumn Bottom-right column index where sub-matrix copy
     *                          ends.
     * @param submatrix         Sub-matrix to be copied
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located below or at right side of bottom-right corner.
     */
    public void setSubmatrix(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn,
            final Matrix submatrix) {
        setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, submatrix, 0,
                0, submatrix.getRows() - 1,
                submatrix.getColumns() - 1);
    }

    /**
     * Copies elements from provided sub-matrix into this matrix at provided
     * location
     *
     * @param topLeftRow                 Top-left row index where sub-matrix copy starts
     * @param topLeftColumn              Top-left column index where sub-matrix copy starts
     * @param bottomRightRow             Bottom-right row index where sub-matrix copy ends
     * @param bottomRightColumn          Bottom-right column index where sub-matrix copy
     *                                   ends.
     * @param submatrix                  Sub-matrix to be copied
     * @param submatrixTopLeftRow        Top-left row index of sub-matrix where copy
     *                                   starts
     * @param submatrixTopLeftColumn     Top-left column index of sub-matrix where
     *                                   copy starts
     * @param submatrixBottomRightRow    Bottom-right row index of sub-matrix where
     *                                   copy ends
     * @param submatrixBottomRightColumn Bottom-right column index of sub-matrix
     *                                   where copy ends
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current or provided matrices, or if
     *                                  top-left corners are indeed located below or at right side of
     *                                  bottom-right corners.
     */
    public void setSubmatrix(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn,
            final Matrix submatrix, final int submatrixTopLeftRow, final int submatrixTopLeftColumn,
            final int submatrixBottomRightRow, final int submatrixBottomRightColumn) {
        if (topLeftRow < 0 || topLeftRow >= rows || topLeftColumn < 0 || topLeftColumn >= columns || bottomRightRow < 0
                || bottomRightRow >= rows || bottomRightColumn < 0 || bottomRightColumn >= columns
                || topLeftRow > bottomRightRow || topLeftColumn > bottomRightColumn) {
            throw new IllegalArgumentException();
        }

        if (submatrixTopLeftRow < 0 || submatrixTopLeftColumn < 0 || submatrixBottomRightRow < 0
                || submatrixBottomRightRow >= submatrix.rows || submatrixBottomRightColumn < 0
                || submatrixBottomRightColumn >= submatrix.columns || submatrixTopLeftRow > submatrixBottomRightRow
                || submatrixTopLeftColumn > submatrixBottomRightColumn) {
            throw new IllegalArgumentException();
        }

        final var matrixRows = bottomRightRow - topLeftRow + 1;
        final var matrixColumns = bottomRightColumn - topLeftColumn + 1;
        final var submatrixRows = submatrixBottomRightRow - submatrixTopLeftRow + 1;
        final var submatrixColumns = submatrixBottomRightColumn - submatrixTopLeftColumn + 1;
        if (matrixRows != submatrixRows || matrixColumns != submatrixColumns) {
            throw new IllegalArgumentException();
        }

        var j2 = submatrixTopLeftColumn;
        int destPos;
        int sourcePos;
        for (var j = topLeftColumn; j <= bottomRightColumn; j++) {
            destPos = columnIndex[j] + topLeftRow;
            sourcePos = submatrix.columnIndex[j2] + submatrixTopLeftRow;
            for (var i = topLeftRow; i <= bottomRightRow; i++) {
                // Lines below are equivalent to commented code
                buffer[destPos] = submatrix.buffer[sourcePos];
                destPos++;
                sourcePos++;
            }
            j2++;
        }
    }

    /**
     * Sets elements in provided region to provided value.
     *
     * @param topLeftRow        Top-left row index of region (inclusive).
     * @param topLeftColumn     Top-left column index of region (inclusive).
     * @param bottomRightRow    Bottom-right row index of region (inclusive).
     * @param bottomRightColumn Bottom-right column index of region (inclusive).
     * @param value             Value to be set.
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located below or at right side of bottom-right corner.
     */
    public void setSubmatrix(
            int topLeftRow, int topLeftColumn, int bottomRightRow, int bottomRightColumn, double value) {
        if (topLeftRow < 0 || topLeftRow >= rows || topLeftColumn < 0 || topLeftColumn >= columns || bottomRightRow < 0
                || bottomRightRow >= rows || bottomRightColumn < 0 || bottomRightColumn >= columns
                || topLeftRow > bottomRightRow || topLeftColumn > bottomRightColumn) {
            throw new IllegalArgumentException();
        }

        for (var j = topLeftColumn; j <= bottomRightColumn; j++) {
            for (var i = topLeftRow; i <= bottomRightRow; i++) {
                buffer[columnIndex[j] + i] = value;
            }
        }
    }

    /**
     * Copies elements from provided array into this matrix at provided
     * location. Elements in array are copied into this matrix considering
     * DEFAULT_USE_COLUMN_ORDER.
     *
     * @param topLeftRow        Top-left row index where copy starts.
     * @param topLeftColumn     Top-left column index where copy starts.
     * @param bottomRightRow    Bottom-right row index where copy ends.
     * @param bottomRightColumn Bottom-right column index where copy
     *                          ends.
     * @param values            Array to be copied.
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located below or at right side of bottom-right corner.
     */
    public void setSubmatrix(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn,
            final double[] values) {
        setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, values, DEFAULT_USE_COLUMN_ORDER);
    }

    /**
     * Copies elements from provided array into this matrix at provided
     * location. Elements in array are copied into this matrix following
     * provided order (either row or column order).
     *
     * @param topLeftRow        Top-left row index where copy starts.
     * @param topLeftColumn     Top-left column index where copy ends.
     * @param bottomRightRow    Bottom-right row index where copy ends.
     * @param bottomRightColumn Bottom-right column index where copy ends.
     * @param values            Array to be copied.
     * @param isColumnOrder     If true values are copied consecutively from array
     *                          following column order on the destination matrix, otherwise row order is
     *                          used.
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located below or at right side of bottom-right corner.
     */
    public void setSubmatrix(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn,
            final double[] values, final boolean isColumnOrder) {
        setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, values, 0,
                values.length - 1, isColumnOrder);
    }

    /**
     * Copies elements from provided array into this matrix at provided
     * location. Elements in array are copied into this matrix considering
     * DEFAULT_USE_COLUMN_ORDER starting at provided location until end provided
     * position.
     *
     * @param topLeftRow        Top-left row index where copy starts (inclusive).
     * @param topLeftColumn     Top-left column index where copy starts (inclusive).
     * @param bottomRightRow    Bottom-right row index where copy ends (inclusive).
     * @param bottomRightColumn Bottom-right column index where copy
     *                          ends. (inclusive)
     * @param values            Array to be copied.
     * @param valuesStart       Position where copy from array will start (inclusive).
     * @param valuesEnd         Position where copy from array will finish (inclusive).
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located below or at right side of bottom-right corner,
     *                                  or if valuesStart and valuesEnd lie outside valid array positions or
     *                                  start is greater than end position.
     */
    public void setSubmatrix(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn,
            final double[] values, final int valuesStart, final int valuesEnd) {
        setSubmatrix(topLeftRow, topLeftColumn, bottomRightRow, bottomRightColumn, values, valuesStart, valuesEnd,
                DEFAULT_USE_COLUMN_ORDER);
    }

    /**
     * Copies elements from provided array into this matrix at provided
     * location. Elements in array are copied into this matrix following
     * provided order (either row or column order) starting at provided location
     * until end provided position.
     *
     * @param topLeftRow        Top-left row index where copy starts (inclusive)
     * @param topLeftColumn     Top-left column index where copy starts (inclusive)
     * @param bottomRightRow    Bottom-right row index where copy ends (inclusive)
     * @param bottomRightColumn Bottom-right column index where copy
     *                          ends. (inclusive)
     * @param values            Array to be copied.
     * @param valuesStart       Position where copy from array will start (inclusive).
     * @param valuesEnd         Position where copy from array will finish (inclusive).
     * @param isColumnOrder     If true values are copied consecutively from array
     *                          following column order on the destination matrix, otherwise row order is
     *                          used.
     * @throws IllegalArgumentException Exception raised whenever top-left or
     *                                  bottom-right corners lie outside current matrix instance, or if top-left
     *                                  corner is indeed located below or at right side of bottom-right corner,
     *                                  or if valuesStart and valuesEnd lie outside valid array positions or
     *                                  start is greater than end position.
     */
    public void setSubmatrix(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn,
            final double[] values, final int valuesStart, final int valuesEnd, final boolean isColumnOrder) {

        if (topLeftRow < 0 || topLeftRow >= rows || topLeftColumn < 0 || topLeftColumn >= columns || bottomRightRow < 0
                || bottomRightRow >= rows || bottomRightColumn < 0 || bottomRightColumn >= columns
                || topLeftRow > bottomRightRow || topLeftColumn > bottomRightColumn) {
            throw new IllegalArgumentException();
        }

        if (valuesStart < 0 || valuesEnd < 0 || valuesEnd >= values.length || valuesStart > valuesEnd) {
            throw new IllegalArgumentException();
        }

        final var matrixRows = bottomRightRow - topLeftRow + 1;
        final var matrixColumns = bottomRightColumn - topLeftColumn + 1;
        final var matrixLength = matrixRows * matrixColumns;
        final var valuesLength = valuesEnd - valuesStart + 1;

        if (matrixLength != valuesLength) {
            throw new IllegalArgumentException();
        }

        var counter = valuesStart;
        if (isColumnOrder) {
            int destPos;
            for (var j = topLeftColumn; j <= bottomRightColumn; j++) {
                destPos = columnIndex[j] + topLeftRow;
                for (var i = topLeftRow; i <= bottomRightRow; i++) {
                    // Two Lines below are equivalent to:
                    // buffer[columnIndex[j] + i] = values[counter]
                    buffer[destPos] = values[counter];
                    destPos++;
                    counter++;
                }
            }

        } else {
            for (var i = topLeftRow; i <= bottomRightRow; i++) {
                for (var j = topLeftColumn; j <= bottomRightColumn; j++) {
                    buffer[columnIndex[j] + i] = values[counter];
                    counter++;
                }
            }
        }
    }

    /**
     * Sets values into provided matrix to make it an identity matrix (all
     * elements in the diagonal equal to one, and remaining elements to zero).
     *
     * @param m Matrix where identity values are set.
     */
    public static void identity(final Matrix m) {
        m.initialize(0.0);

        final var minSize = Math.min(m.rows, m.columns);
        for (var i = 0; i < minSize; i++) {
            m.buffer[m.columnIndex[i] + i] = 1.0;
        }
    }

    /**
     * Creates and returns a new matrix instance having all the elements on
     * the diagonal equal to one and the remaining ones equal to zero.
     *
     * @param rows    Number of rows of created instance
     * @param columns Number of columns of created instance
     * @return An identity matrix
     * @throws WrongSizeException Raised if either rows or columns is
     *                            equal to zero
     */
    public static Matrix identity(final int rows, final int columns) throws WrongSizeException {
        final var out = new Matrix(rows, columns);
        identity(out);
        return out;
    }

    /**
     * Fills provided matrix with random uniform values ranging from minValue to
     * maxValue.
     *
     * @param minValue Minimum value of uniform random generated values
     * @param maxValue Maximum value of uniform random generated values
     * @param random   Random generator
     * @param result   Matrix where random values are stored.
     * @throws IllegalArgumentException if minValue &lt;= maxValue
     * @throws NullPointerException     if either provided random or result
     *                                  instances are null
     */
    public static void fillWithUniformRandomValues(
            final double minValue, final double maxValue, final Random random, final Matrix result) {
        final var randomizer = new UniformRandomizer(random);

        final var length = result.rows * result.columns;

        for (int i = 0; i < length; i++) {
            result.buffer[i] = randomizer.nextDouble(minValue, maxValue);
        }
    }

    /**
     * Fills provided matrix with random uniform values ranging from minValue to
     * maxValue.
     *
     * @param minValue Minimum value of uniform random generated values
     * @param maxValue Maximum value of uniform random generated values
     * @param result   Matrix where random values are stored
     * @throws IllegalArgumentException if minValue &lt;= maxValue
     * @throws NullPointerException     if provided result matrix is null
     */
    public static void fillWithUniformRandomValues(final double minValue, final double maxValue, final Matrix result) {
        fillWithUniformRandomValues(minValue, maxValue, new Random(), result);
    }

    /**
     * Creates new matrix instance using provided size and containing uniformly
     * distributed random values with provided range.
     *
     * @param rows     Number of rows of instantiated matrix
     * @param columns  Number of columns of instantiated matrix
     * @param minValue Minimum value of uniform random generated values
     * @param maxValue Maximum value of uniform random generated values
     * @return A new matrix containing uniform random values
     * @throws WrongSizeException       Exception thrown if either rows or
     *                                  columns is zero, or if the minimum random value is greater or equal
     *                                  than the maximum random value
     * @throws IllegalArgumentException if minValue &lt;= maxValue
     */
    public static Matrix createWithUniformRandomValues(
            final int rows, final int columns, final double minValue, final double maxValue) throws WrongSizeException {
        return createWithUniformRandomValues(rows, columns, minValue, maxValue, new Random());
    }

    /**
     * Creates new matrix instance using provided size and containing uniformly
     * distributed random values with provided range and using provided random
     * generator
     *
     * @param rows     Number of rows of instantiated matrix
     * @param columns  Number of columns of instantiated matrix
     * @param minValue Minimum value of uniform random generated values
     * @param maxValue Maximum value of uniform random generated values
     * @param random   A random generator.
     * @return A new matrix containing uniform random values
     * @throws WrongSizeException       Exception thrown if either rows or
     *                                  columns is zero, or if the minimum random value is greater or equal
     *                                  than the maximum random value
     * @throws IllegalArgumentException if minValue &lt;= maxValue
     */
    public static Matrix createWithUniformRandomValues(
            final int rows, final int columns, final double minValue, final double maxValue, final Random random)
            throws WrongSizeException {

        final var out = new Matrix(rows, columns);
        fillWithUniformRandomValues(minValue, maxValue, random, out);
        return out;
    }

    /**
     * Fills provided matrix with random Gaussian values with provided mean and
     * standard deviation.
     *
     * @param mean              Mean value of generated random values
     * @param standardDeviation Standard deviation of generated random values
     * @param random            Random generator
     * @param result            Matrix where random values are stored
     * @throws IllegalArgumentException if standard deviation is negative or
     *                                  zero
     * @throws NullPointerException     if provided result matrix is null
     */
    public static void fillWithGaussianRandomValues(
            final double mean, final double standardDeviation, final Random random, final Matrix result) {

        final var randomizer = new GaussianRandomizer(random, mean, standardDeviation);

        final var length = result.rows * result.columns;

        for (var i = 0; i < length; i++) {
            result.buffer[i] = randomizer.nextDouble();
        }
    }

    /**
     * Fills provided matrix with random Gaussian values with provided mean and
     * standard deviation
     *
     * @param mean              Mean value of generated random values
     * @param standardDeviation Standard deviation of generated random values
     * @param result            Matrix where random values are stored
     * @throws IllegalArgumentException if standard deviation is negative or
     *                                  zero
     * @throws NullPointerException     if provided result matrix is null
     */
    public static void fillWithGaussianRandomValues(
            final double mean, final double standardDeviation, final Matrix result) {
        fillWithGaussianRandomValues(mean, standardDeviation, new Random(), result);
    }

    /**
     * Creates new matrix instance using provided size and containing
     * gaussian/normal distributed random values with provided median and
     * standard deviation.
     *
     * @param rows              Number of rows of instantiated matrix
     * @param columns           Number of columns of instantiated matrix
     * @param mean              Mean value of gaussian random generated values
     * @param standardDeviation Standard deviation of gaussian random generated
     *                          values
     * @return A new matrix containing gaussian random values
     * @throws WrongSizeException       Exception thrown if either rows or
     *                                  columns is zero, or if the standard deviation is negative or zero.
     * @throws IllegalArgumentException thrown if provided standard deviation
     *                                  is negative or zero.
     */
    public static Matrix createWithGaussianRandomValues(
            final int rows, final int columns, final double mean, final double standardDeviation)
            throws WrongSizeException {
        return createWithGaussianRandomValues(rows, columns, mean, standardDeviation, new Random());
    }

    /**
     * Creates new matrix instance using provided size and containing
     * gaussian/normal distributed random values with provided median and
     * standard deviation and using provided random generator
     *
     * @param rows              Number of rows of instantiated matrix
     * @param columns           Number of columns of instantiated matrix
     * @param mean              Mean value of gaussian random generated values
     * @param standardDeviation Standard deviation of gaussian random generated
     *                          values
     * @param random            A random generator.
     * @return A new matrix containing gaussian random values
     * @throws WrongSizeException       Exception thrown if either rows or
     *                                  columns is zero, or if the standard deviation is negative or zero.
     * @throws IllegalArgumentException thrown if provided standard deviation
     *                                  is negative or zero.
     */
    public static Matrix createWithGaussianRandomValues(
            final int rows, final int columns, final double mean, final double standardDeviation, final Random random)
            throws WrongSizeException {

        final var out = new Matrix(rows, columns);
        fillWithGaussianRandomValues(mean, standardDeviation, random, out);
        return out;
    }

    /**
     * Makes provided result matrix a diagonal matrix containing provided
     * elements in the diagonal. Elements outside the diagonal will be set to
     * zero
     *
     * @param diagonal Array containing the elements to be set on the diagonal
     * @param result   Matrix where values are stored
     * @throws NullPointerException Exception thrown if provided parameters are
     *                              null
     */
    public static void diagonal(final double[] diagonal, final Matrix result) {

        result.initialize(0.0);
        // set diagonal elements
        //noinspection ManualArrayCopy
        for (var i = 0; i < diagonal.length; i++) {
            result.buffer[result.columnIndex[i] + i] = diagonal[i];
        }
    }

    /**
     * Creates a diagonal matrix having all the elements in provided array in
     * its diagonal and the remaining elements equal to zero.
     * Returned matrix will have size n x n, where n is the length of the array
     *
     * @param diagonal Array containing the elements to be set on the diagonal
     * @return A diagonal matrix
     * @throws NullPointerException Raised if provided diagonal array is null
     */
    public static Matrix diagonal(final double[] diagonal) {

        Matrix out = null;
        try {
            out = new Matrix(diagonal.length, diagonal.length);
            diagonal(diagonal, out);
        } catch (final WrongSizeException ignore) {
            // never happens
        }

        return out;
    }

    /**
     * Instantiates new matrix from array using DEFAULT_USE_COLUMN_ORDER
     *
     * @param array Array used as source to copy values from
     * @return Returns matrix created from array
     */
    public static Matrix newFromArray(final double[] array) {
        return newFromArray(array, DEFAULT_USE_COLUMN_ORDER);
    }

    /**
     * Instantiates new matrix from array using either column or row order
     *
     * @param array         Array used as source to copy values from
     * @param isColumnOrder True if column order must be used, false otherwise
     * @return Returns matrix created from array
     */
    public static Matrix newFromArray(final double[] array, final boolean isColumnOrder) {
        Matrix m = null;
        try {
            if (isColumnOrder) {
                m = new Matrix(array.length, 1);
                m.setSubmatrix(0, 0, array.length - 1, 0, array);
            } else {
                m = new Matrix(1, array.length);
                m.setSubmatrix(0, 0, 0, array.length - 1, array);
            }
        } catch (final WrongSizeException ignore) {
            // never happens
        }
        return m;
    }

    /**
     * Copies elements of array into this instance using column order.
     *
     * @param array array to copy values from.
     * @throws WrongSizeException if provided array length is not equal to the
     *                            number of rows multiplied per the number of columns of this instance.
     */
    public void fromArray(final double[] array) throws WrongSizeException {
        fromArray(array, DEFAULT_USE_COLUMN_ORDER);
    }

    /**
     * Copies elements of array into this instance using provided order.
     *
     * @param array         array to copy values from.
     * @param isColumnOrder true to use column order, false otherwise.
     * @throws WrongSizeException if provided array length is not equal to the
     *                            number of rows multiplied per the number of columns of this instance.
     */
    public void fromArray(final double[] array, final boolean isColumnOrder) throws WrongSizeException {
        if (array.length != buffer.length) {
            throw new WrongSizeException("array length must be equal to rows x columns");
        }

        if (isColumnOrder) {
            System.arraycopy(array, 0, buffer, 0, array.length);
        } else {
            var counter = 0;

            for (var j = 0; j < rows; j++) {
                for (var i = 0; i < columns; i++) {
                    buffer[columnIndex[i] + j] = array[counter];
                    counter++;
                }
            }
        }
    }

    /**
     * Symmetrizes this instance and stores the result into provided instance.
     * Symmetrization is done by averaging this instance with its transpose
     * (i.e. S = (M+M')/2
     *
     * @param result instance where symmetrized version of this instance will be
     *               stored.
     * @throws WrongSizeException if this instance is not square or provided
     *                            result instance doesn't have the same size as this instance.
     */
    public void symmetrize(final Matrix result) throws WrongSizeException {
        if (rows != columns) {
            throw new WrongSizeException("matrix must be square");
        }
        if (result.getRows() != rows || result.getColumns() != columns) {
            throw new WrongSizeException("result matrix must have the size of this instance");
        }

        // S = (M+M')/2
        double value1;
        double value2;
        double avg;
        int pos1;
        int pos2;
        for (var i = 0; i < columns; i++) {
            for (var j = i; j < rows; j++) {
                // value at (i, j)
                pos1 = columnIndex[i] + j;
                value1 = buffer[pos1];
                // transposed value (i.e. value at (j,i))
                pos2 = columnIndex[j] + i;
                value2 = buffer[pos2];

                avg = 0.5 * (value1 + value2);
                result.buffer[pos1] = avg;
                result.buffer[pos2] = avg;
            }
        }
    }

    /**
     * Symmetrizes this instance and returns the result as a new matrix
     * instance.
     * Symmetrization is done by averaging this instance with its transpose
     * (i.e. S = (M+M')/2
     *
     * @return a new symmetrizes version of this instance.
     * @throws WrongSizeException if this instance is not square.
     */
    public Matrix symmetrizeAndReturnNew() throws WrongSizeException {
        final var m = new Matrix(rows, columns);
        symmetrize(m);
        return m;
    }

    /**
     * Symmetrizes this instance and updates it with computed value.
     * Symmetrization is done by averaging this instance with its transpose
     * (i.e. S = (M+M')/2
     *
     * @throws WrongSizeException if this instance is not square.
     */
    public void symmetrize() throws WrongSizeException {
        symmetrize(this);
    }

    /**
     * Method to internally add two matrices.
     *
     * @param other  Matrix to be added to current matrix
     * @param result Matrix where result will be stored.
     */
    private void internalAdd(final Matrix other, final Matrix result) {
        final var length = rows * columns;
        for (var i = 0; i < length; i++) {
            result.buffer[i] = buffer[i] + other.buffer[i];
        }
    }

    /**
     * Method to internally subtract two matrices.
     *
     * @param other  Matrix to be subtracted from current matrix.
     * @param result Matrix where result will be stored.
     */
    private void internalSubtract(final Matrix other, final Matrix result) {
        final var length = rows * columns;
        for (var i = 0; i < length; i++) {
            result.buffer[i] = buffer[i] - other.buffer[i];
        }
    }

    /**
     * Method to internally multiply two matrices.
     *
     * @param other             Matrix to be multiplied to current matrix
     * @param resultBuffer      Matrix buffer of data where result will be stored.
     * @param resultColumnIndex Array of matrix column indices where result will
     *                          be stored.
     */
    private void internalMultiply(
            final Matrix other, final double[] resultBuffer, final int[] resultColumnIndex) {
        final var columns2 = other.columns;
        double value;
        for (var k = 0; k < columns2; k++) {
            for (var j = 0; j < rows; j++) {
                value = 0.0;
                for (var i = 0; i < columns; i++) {
                    value += buffer[columnIndex[i] + j] *
                            other.buffer[other.columnIndex[k] + i];
                }
                resultBuffer[resultColumnIndex[k] + j] = value;
            }
        }
    }

    /**
     * Method to internally multiply two matrices.
     *
     * @param other  Matrix to be multiplied to current matrix.
     * @param result Matrix where result will be stored.
     */
    private void internalMultiply(final Matrix other, final Matrix result) {
        internalMultiply(other, result.buffer, result.columnIndex);
    }

    /**
     * Method to internally compute the Kronecker product between two matrices.
     *
     * @param other             other matrix to be Kronecker multiplied to current matrix.
     * @param resultBuffer      matrix buffer of data where result will be stored.
     * @param resultColumnIndex array of matrix column indices where result will
     *                          be stored.
     */
    private void internalMultiplyKronecker(
            final Matrix other, final double[] resultBuffer, final int[] resultColumnIndex) {
        final var rows2 = other.rows;
        final var columns2 = other.columns;

        for (var j1 = 0; j1 < rows; j1++) {
            final var startJ3 = j1 * other.rows;
            for (var i1 = 0; i1 < columns; i1++) {
                final var startI3 = i1 * other.columns;
                final var value1 = buffer[columnIndex[i1] + j1];

                for (var j2 = 0; j2 < rows2; j2++) {
                    final var j3 = startJ3 + j2;
                    for (var i2 = 0; i2 < columns2; i2++) {
                        final var i3 = startI3 + i2;
                        final var value2 = other.buffer[other.columnIndex[i2] + j2];

                        final var value3 = value1 * value2;
                        resultBuffer[resultColumnIndex[i3] + j3] = value3;
                    }
                }
            }
        }
    }

    /**
     * Method to internally compute the Kronecker product between two matrices.
     *
     * @param other  other matrix to be Kronecker multiplied to current matrix.
     * @param result matrix where result will be stored.
     */
    private void internalMultiplyKronecker(final Matrix other, final Matrix result) {
        internalMultiplyKronecker(other, result.buffer, result.columnIndex);
    }

    /**
     * Method to internally compute element by element product of two matrices.
     *
     * @param other  Matrix to be element by element multiplied to current matrix
     * @param result Matrix where result will be stored.
     */
    private void internalElementByElementProduct(final Matrix other, final Matrix result) {
        final var length = rows * columns;
        for (var i = 0; i < length; i++) {
            result.buffer[i] = buffer[i] * other.buffer[i];
        }
    }

    /**
     * Method to internally compute matrix transposition.
     *
     * @param resultBuffer      Buffer where transposed matrix data is stored.
     * @param resultColumnIndex Buffer where indices of transposed matrix data
     *                          is stored.
     */
    private void internalTranspose(final double[] resultBuffer, final int[] resultColumnIndex) {
        for (var j = 0; j < rows; j++) {
            for (var i = 0; i < columns; i++) {
                resultBuffer[resultColumnIndex[j] + i] = buffer[columnIndex[i] + j];
            }
        }
    }

    /**
     * Method to internally compute matrix transposition.
     *
     * @param result Matrix where transposed data is stored.
     */
    private void internalTranspose(final Matrix result) {
        internalTranspose(result.buffer, result.columnIndex);
    }

    /**
     * Method used internally to remove matrix contents and resizing it.
     *
     * @param rows    Number of rows to be set
     * @param columns Number of columns to be set.
     * @throws WrongSizeException Exception raised if either rows or
     *                            columns is zero.
     */
    private void internalResize(final int rows, final int columns) throws WrongSizeException {
        if (rows == 0 || columns == 0) {
            throw new WrongSizeException();
        }

        this.rows = rows;
        this.columns = columns;

        // instantiate buffers of data
        buffer = new double[rows * columns];
        columnIndex = new int[columns];

        // initialize column index
        var counter = 0;
        for (var i = 0; i < columns; i++) {
            columnIndex[i] = counter;
            counter += rows;
        }
    }

    /**
     * Internal method to retrieve a sub-matrix of current matrix instance.
     * Sub-matrix is obtained by copying all elements contained within provided
     * coordinates (both top-left and bottom-right points are included within
     * sub-matrix).
     *
     * @param topLeftRow        Top-left row index where sub-matrix starts.
     * @param topLeftColumn     Top-left column index where sub-matrix starts.
     * @param bottomRightRow    Bottom-right row index where sub-matrix ends.
     * @param bottomRightColumn Bottom-right column index where sub-matrix ends.
     * @param result            Instance where sub-matrix data is stored.
     */
    private void internalGetSubmatrix(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow, final int bottomRightColumn,
            final Matrix result) {
        var i2 = 0;
        var j2 = 0;
        for (var j = topLeftColumn; j <= bottomRightColumn; j++) {
            for (var i = topLeftRow; i <= bottomRightRow; i++) {
                result.buffer[result.columnIndex[j2] + i2] = buffer[columnIndex[j] + i];
                i2++;
            }
            i2 = 0;
            j2++;
        }
    }

    /**
     * Internal method to retrieve a sub-matrix of current matrix instance as an
     * array of values using provided column order and storing the result in
     * provided array.
     * Sub-matrix is obtained by copying all elements contained within provided
     * coordinates (both top-left and bottom-right points are included within
     * sub-matrix).
     *
     * @param topLeftRow        Top-left row index where sub-matrix starts
     * @param topLeftColumn     Top-left column index where sub-matrix starts
     * @param bottomRightRow    Bottom-right row index where sub-matrix ends
     * @param bottomRightColumn Bottom-right column index where sub-matrix ends.
     * @param isColumnOrder     If true, picks elements from matrix using column
     *                          order, otherwise row order is used.
     * @param result            Array where sub-matrix data is stored.
     */
    private void internalGetSubmatrixAsArray(
            final int topLeftRow, final int topLeftColumn, final int bottomRightRow,
            final int bottomRightColumn, final boolean isColumnOrder, final double[] result) {
        var counter = 0;
        if (isColumnOrder) {
            for (var j = topLeftColumn; j <= bottomRightColumn; j++) {
                for (var i = topLeftRow; i <= bottomRightRow; i++) {
                    result[counter] = buffer[columnIndex[j] + i];
                    counter++;
                }
            }
        } else {
            for (var i = topLeftRow; i <= bottomRightRow; i++) {
                for (var j = topLeftColumn; j <= bottomRightColumn; j++) {
                    result[counter] = buffer[columnIndex[j] + i];
                    counter++;
                }
            }
        }
    }
}
