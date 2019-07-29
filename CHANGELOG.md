# Change Log

All notable changes to this project will be documented in this file, following
the format defined at [keepachangelog.com](http://keepachangelog.com/).
This project adheres to [Semantic Versioning](http://semver.org/) as of version 0.3.

## [v0.3.0]

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

