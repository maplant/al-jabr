# Change Log

All notable changes to this project will be documented in this file, following
the format defined at [keepachangelog.com](http://keepachangelog.com/).
This project adheres to [Semantic Versioning](http://semver.org/) as of version 0.3.

## Unreleased Changes
These changes are included in `master` but have not been released in a new crate version yet.

There are no unreleased changes at the current moment.

## [v0.4.2] - 2019-12-07

- Add support for the `mint` crate.
- Add `IntoIterato` implementations for `Vector`, `Matrix` and `Point`.

## [v0.4.1] - 2019-09-20

- Fix a typo in the documentation.

## [v0.4.0] - 2019-09-20

- Add `FromIterator` implementation for `Vector` and `Matrix`.
- Remove redundant `SquareMatrix` requirement from  `Matrix` impl of `One`.
- Implement `invert`  for Matrices up to dimension of 2.
- Add `One`, `Div<Self, Output = Self>` and `Neg<Output = Self>` constraints 
  to `SquareMatrix::Scalar`.

## [v0.3.2] - 2019-08-29

- Remove `trunc` method and deprecated `TruncatedVector` due to an ice.
- Added `reflect` method simply because it was already in master. Sorry semver.
- Change license to dual MIT/Apache-2.0

## [v0.3.1] 

- Fix some typos in the docs.

## [v0.3.0] - 2019-07-28

- Add `vector!` and `matrix!` macros, deprecate other construction methods.
- Add `Point` type.
- Add `map` method to `Vector` and `Matrix`.
- Implement `Distribution<Vector<_, _>>` and `Distribution<Matrix<_, _, _>>` for
  rand `Standard`.
- Implement serde `Serialize` and `Deserialize` for all data structures.
- Add `Rotation<DIMS>` trait to describe values that can rotate vectors of size `DIMS`.
- Add `Angle` trait to describe values with a `sin` and `cos` defined. 
- Add `Euler` struct to describe rotations in three dimensions via three components. 
- Add `Orthonormal` struct for rotation matrices. 

