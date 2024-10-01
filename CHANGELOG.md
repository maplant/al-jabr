# Change Log

All notable changes to this project will be documented in this file, following
the format defined at [keepachangelog.com](http://keepachangelog.com/).
This project adheres to [Semantic Versioning](http://semver.org/) as of version 0.3.

## Unreleased Changes
These changes are included in `master` but have not been released in a new crate version yet.

## [v0.6.0]
- Added `Vector2`, `Vector3` and `Vector4` structs to improve ergonomics while dealing with smaller
  vector types.
- Added `Rotation2` and `Rotation3` traits for rotating small vectors and points.
- Changed `Vector{N}` to `ColumnVector{N}`.
- Changed `Point` types be struct based rather than array based to reflect the change in vectors.
- Changed `Rotation` to `RotationN`

## [v0.5.6]
- Added some missing derives to `Euler` and `Unit`.

## [v0.5.5]
- Added `to_scale_rotation_translation` function to Matrix4.
- Added `From<Orthonormal<T, 3>>` for `Quaternion<T>`.
- Added `new` function to `Orthonormal`.
- Added `signum` function to `Real`.

## [v0.5.4]
- Fix affine matrix construction.

## [v0.5.3]

- Added `from_rotation` constructor for `Matrix3` and `Matrix4`.
- Derive `serde::Serialize` and `serde::Serialize` and for `Unit` when `T` supports it.

## [v0.5.2]

- Added `Vector1` and `Point1`
- Added `from_vec1`, `from_vec2`, and `from_vec3` constructors to extend vectors.
- Added `from_point1`, `from_point1`, and `from_point1`, constructors to extend points.
- Make `Unit` `Copy` and `Clone` dependent on `T`.

## [v0.5.1]

- Remove redundant `lerp` definition (oops).

## [v0.5.0]

- Added `asin` and `acos` methods to `Real`.
- Added `Unit` struct to enforce normalized objects.
- Added `nlerp` and `slerp` method to `Unit`.
- Added `lerp` method to `Matrix`.

## [v0.4.1] 

- Implement remaining `approx` traits for `Matrix` and `Point`.
- Change to edition 2021.

## [v0.2.0]

- Redefine Vector in terms of Matrices, as opposed to definining Matrices in terms of Vectors.
  This is a more natural definition and allows for a dramatic reduction of code.

## [v0.1.0] 

- Renamed the crate to `al-jabr` and remove methods that are not compatible with rust stable.

## [v1.0.2] - 2020-10-26

- Added `const_evaluatable_checked` unstable feature to allow for `truncate` and `extend` methods to be used.

## [v1.0.1] - 2020-07-19

- Fix an instance of undefined behavior in `swap_columns` and `swap_rows` when attempting to 
  swap a row or column with itself.

## [v1.0.0] - 2020-05-29

- Rename `trunc` method `truncate`.
- Added `extend`, `max`, `argmax`, `min`, and `argmin` to `Vector`.
- Added `column_iter`, `column_iter_mut`, `row_iter`, `row_iter_mut` to `Matrix`.
- Added `LU` matrix.
- Added complete `inverse`, `determinant` and `lu` methods to `Matrix`.
- Remove `Angle` trait.
- Remove `SquareMatrix` trait. 
- Reorganize `Matrix` type aliases. 

## [v0.5.0] - 2020-04-25

- Added `trunc` method to `Vector`, now that rust supports it. 
- Remove deprecation from `TruncatedVector`. 

## [v0.4.2] - 2019-12-07

- Added support for the `mint` crate.
- Added `IntoIterator` implementations for `Vector`, `Matrix` and `Point`.

## [v0.4.1] - 2019-09-20

- Fix a typo in the documentation.

## [v0.4.0] - 2019-09-20

- Added `FromIterator` implementation for `Vector` and `Matrix`.
- Remove redundant `SquareMatrix` requirement from  `Matrix` impl of `One`.
- Implement `invert`  for Matrices up to dimension of 2.
- Added `One`, `Div<Self, Output = Self>` and `Neg<Output = Self>` constraints 
  to `SquareMatrix::Scalar`.

## [v0.3.2] - 2019-08-29

- Remove `trunc` method and deprecated `TruncatedVector` due to an ice.
- Added `reflect` method simply because it was already in master. Sorry semver.
- Change license to dual MIT/Apache-2.0

## [v0.3.1] 

- Fix some typos in the docs.

## [v0.3.0] - 2019-07-28

- Added `vector!` and `matrix!` macros, deprecate other construction methods.
- Added `Point` type.
- Added `map` method to `Vector` and `Matrix`.
- Implement `Distribution<Vector<_, _>>` and `Distribution<Matrix<_, _, _>>` for
  rand `Standard`.
- Implement serde `Serialize` and `Deserialize` for all data structures.
- Added `Rotation<DIMS>` trait to describe values that can rotate vectors of size `DIMS`.
- Added `Angle` trait to describe values with a `sin` and `cos` defined. 
- Added `Euler` struct to describe rotations in three dimensions via three components. 
- Added `Orthonormal` struct for rotation matrices. 

