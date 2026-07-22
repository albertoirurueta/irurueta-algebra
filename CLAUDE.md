# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`irurueta-algebra` is a small Java 21 linear algebra library (Maven, `com.irurueta:irurueta-algebra`), published to Maven Central. It provides a dense `Matrix` type, matrix decompositions (LU, QR, economy QR, RQ, Cholesky, SVD), norm computation, complex numbers, and general-purpose algebra helpers in `Utils`/`ArrayUtils`. It depends on the sibling library `com.irurueta:irurueta-statistics` and is itself a building block for other `irurueta-*` libraries (e.g. geometry, numerical methods).

## Commands

Build and test (default profile runs Checkstyle-independent build; Checkstyle/PMD/SpotBugs/Javadoc reports are Maven `reporting` plugins, not bound to `verify`):

```bash
mvn clean install                         # compile, run tests, install to local repo
mvn test                                  # run the full unit test suite
mvn test -Dtest=MatrixTest                # run a single test class
mvn test -Dtest=MatrixTest#testMultiply   # run a single test method
mvn test -Dtest="LUDecomposer*"           # run tests matching a pattern

mvn clean jacoco:prepare-agent test jacoco:report   # unit tests + JaCoCo coverage report (target/site/jacoco)
mvn checkstyle:checkstyle                 # style report (target/checkstyle-result.xml), rules in checkstyle.xml
mvn pmd:pmd spotbugs:spotbugs             # static analysis reports (target/pmd.xml, target/spotbugsXml.xml)
mvn site                                  # full reporting site (javadoc, coverage, checkstyle, pmd, spotbugs, jxr)
```

Notes:
- Tests use JUnit 5 (Jupiter); test classes are package-private (`class FooTest`), matching production package structure under `src/test/java/com/irurueta/algebra` and `.../statistics`.
- The `extras` Maven profile (active by default) attaches sources and Javadoc jars; disable it with `-P '!extras'` for plain builds (as CI does).
- The `sign` profile (GPG signing) and deploy to Maven Central require credentials and are only relevant to release CI, not local development.
- A `validate`-phase build step (via `groovy-maven-plugin`) regenerates `src/main/resources/com/irurueta/algebra/build-info.properties` (backing `BuildInfo.java`) on every Maven invocation — this is expected, not a stray diff to "fix", though it can be a spurious uncommitted change worth checking before committing.

## Architecture

**`Matrix`** (`Matrix.java`) is the central data type: a dense, row/column `double[]`-backed matrix (`Serializable`, `Cloneable`) supporting element access, arithmetic (`add`/`subtract`/`multiply`, both in-place and `*AndReturnNew` variants), Kronecker/element-wise products, transposition, submatrix extraction, and static factories (`identity`, `diagonal`, `createWithUniformRandomValues`, `createWithGaussianRandomValues`, `newFromArray`).

**Decomposers** (`Decomposer.java` abstract base + `LUDecomposer`, `QRDecomposer`, `EconomyQRDecomposer`, `RQDecomposer`, `CholeskyDecomposer`, `SingularValueDecomposer`) all share the same lifecycle contract rather than a shared factory:
- Constructed with (or later given via `setInputMatrix`) the `Matrix` to decompose.
- `isReady()` — has an input matrix been set.
- `isLocked()` / `locked` field — sublcasses set this while `decompose()` runs; setters throw `LockedException` while locked.
- `decompose()` — performs the decomposition; throws `NotReadyException`, `LockedException`, or `DecomposerException` (or subclasses like `NoConvergenceException`, `NonSymmetricPositiveDefiniteMatrixException`).
- `isDecompositionAvailable()` guards subsequent result getters, which otherwise throw `NotAvailableException`.
- `DecomposerType` enumerates the algorithms; there is no `Decomposer.create(type)` factory — callers instantiate the concrete decomposer class directly.

**Norm computers** (`NormComputer.java` abstract base + `FrobeniusNormComputer`, `InfinityNormComputer`, `OneNormComputer`) *do* follow a factory pattern: `NormComputer.create(NormType)` / `NormComputer.create()` (defaults to `NormType.FROBENIUS_NORM`).

**`Utils`** is the high-level static-method entry point most consumers use instead of decomposers directly: `det`, `rank`, `cond`, `solve` (matrix or array right-hand side), `inverse`/`pseudoInverse`, `norm1`/`norm2`/`normF`/`normInf`, `skewMatrix`, `crossProduct`, `isSymmetric`, `trace`. Internally these pick and drive the appropriate `Decomposer`/`NormComputer`. `ArrayUtils` holds the equivalent lower-level array-based numeric helpers.

**`Complex`** implements complex-number arithmetic used by `SingularValueDecomposer`/eigen-related code paths.

**Exception hierarchy**: `AlgebraException` is the common checked-exception root; most public algebra methods throw one of its subclasses (`WrongSizeException`, `SingularMatrixException`, `RankDeficientMatrixException`, `DecomposerException` and its subclasses, `NotReadyException`, `NotAvailableException`, `LockedException`) rather than unchecked exceptions — check a method's `throws` clause to know which of these can surface.

**`com.irurueta.statistics`** (small, separate from the main `algebra` package): `MultivariateNormalDist` and `MultivariateGaussianRandomizer`, plus `InvalidCovarianceMatrixException` — used together with `Matrix`/decomposers for Gaussian sampling and covariance validation.

## Code style

- `checkstyle.xml` enforces (among others): 120-char line length, no tabs, mandatory Javadoc on every package (`JavadocPackage` — hence a `package-info.java` per package), braces required on all blocks, `HiddenField`, `MagicNumber`, max 2 declared exception types per `throws` clause (`ThrowsCount`), and `MissingOverride`/`MissingDeprecated` annotation checks.
- Constructor/method parameters are consistently declared `final`.
- Every public/protected member carries a Javadoc block (see `Decomposer.java` for the house style: full explanation of state transitions and `@throws` semantics per method).
