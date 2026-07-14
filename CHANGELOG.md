# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.0] - 2026-07-06

### Added

- Antora documentation site (`docs/`) with dedicated pages for each decomposition algorithm (LU, QR, economy QR,
  RQ, Cholesky, SVD), each norm computer (Frobenius, infinity, one-norm), Gauss-Jordan elimination, installation,
  and an API reference, published to GitHub Pages alongside the existing Maven site report.

### Changed

- Rewrote `README.md`.
- Updated CI workflows (`develop.yml`, `master.yml`) to build and publish the Antora site.
- Updated the `irurueta-statistics` dependency to 1.4.0.

No changes were made to the library's public API or runtime behavior in this release.

## [1.3.2] - 2025-09-22

### Removed

- Removed the stray `package-info.java` from the `com.irurueta.statistics` package.

## [1.3.1] - 2025-09-18

### Changed

- Removed the unused Maven Central snapshot `distributionManagement`/`repositories` sections from `pom.xml`.
- Updated JUnit Jupiter to 5.13.4, `irurueta-statistics` to 1.3.4, and other build plugin versions (GPG plugin
  and related tooling).
- Updated GitHub Actions workflows (`develop.yml`, `manual_develop.yml`, `master.yml`).

## [1.3.0] - 2024-10-19

### Changed

- Migrated the project from Java 7 to Java 17 and adopted modern language features (`var` type inference,
  pattern-matching `instanceof`).
- Migrated the test suite from JUnit 4 to JUnit 5.
- Replaced the unmaintained `findbugs-maven-plugin` with `spotbugs-maven-plugin`, and updated the
  Surefire/Failsafe/JaCoCo/Checkstyle/PMD/JXR/site plugin versions.
- Updated the `irurueta-statistics` dependency to 1.3.2.
- Reformatted source code to a consistent 120-character line length.

No changes were made to the library's public API in this release.

## [1.2.0] - 2023-11-17

### Changed

- Updated the `irurueta-statistics` dependency to 1.2.0.
- Minor Javadoc wording fixes and build tooling touch-ups.

No changes were made to the library's public API in this release.

## [1.1.0] - 2021-12-11

### Added

- Initial release of the algebra library.
- Dense `Matrix` type with element access, arithmetic operations (add, subtract, multiply, Kronecker and
  element-wise products, transposition), submatrix extraction, and factory methods (identity, diagonal, random
  values, from array).
- `Complex` number arithmetic.
- Matrix decomposition algorithms: LU, QR, economy QR, RQ, Cholesky, and Singular Value Decomposition (SVD), plus
  Gauss-Jordan elimination.
- Norm computation (Frobenius, infinity, one-norm) via a `NormComputer` factory.
- High-level `Utils`/`ArrayUtils` helpers for determinant, rank, condition number, linear system solving, matrix
  inversion/pseudo-inverse, cross product, skew matrix, and symmetry checks.
- `com.irurueta.statistics` package with `MultivariateNormalDist` and `MultivariateGaussianRandomizer` for
  Gaussian sampling and covariance validation.
- Dedicated checked-exception hierarchy (`AlgebraException` and its subclasses) for algebra error conditions.

[Unreleased]: https://github.com/albertoirurueta/irurueta-algebra/compare/1.4.0...HEAD
[1.4.0]: https://github.com/albertoirurueta/irurueta-algebra/compare/1.3.2...1.4.0
[1.3.2]: https://github.com/albertoirurueta/irurueta-algebra/compare/1.3.1...1.3.2
[1.3.1]: https://github.com/albertoirurueta/irurueta-algebra/compare/1.3.0...1.3.1
[1.3.0]: https://github.com/albertoirurueta/irurueta-algebra/compare/1.2.0...1.3.0
[1.2.0]: https://github.com/albertoirurueta/irurueta-algebra/compare/1.1.0...1.2.0
[1.1.0]: https://github.com/albertoirurueta/irurueta-algebra/releases/tag/1.1.0
