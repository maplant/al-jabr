# aljabar 

[![Documentation](https://docs.rs/aljabar/badge.svg)](https://docs.rs/aljabar)
[![Version](https://img.shields.io/crates/v/aljabar.svg)](https://crates.io/crates/aljabar)
[![Downloads](https://img.shields.io/crates/d/aljabar.svg)](https://crates.io/crates/aljabar)

An experimental n-dimensional linear algebra and mathematics library for computer
graphics, designed to be roughly compatible with [cgmath](https://github.com/rustgd/cgmath).

The library provides:

* vectors: `Vector2`, `Vector3`, `Vector4` and `Vector<T, const N: usize>`
* points: `Point2`, `Point3`, `Point4` and `Point<T, const N: usize>`
* matrices: `Matrix2`, `Matrix3`, `Matrix4` and `Matrix<T, const N: usize, const M: usize>`
* a quaternion type: `Quaternion`
* orthonormal (rotation) matrices: `Orthonormal`


`aljabar` supports Vectors and Matrices of any size and will provide 
implementations for any mathematic operations that are supported by their
scalars. Additionally, aljabar can leverage Rust's type system to ensure that
operations are only applied to values that are the correct size.

`aljabar` relies heavily on unstable Rust features such as const generics and thus
requires nightly to build. 

For more information and a guide on getting started, check out the [documentation](https://docs.rs/aljabar/).

## Cargo Features

* The `mint` feature (off by default) adds a dependency to the [mint](https://crates.io/crates/mint) crate and provides support for converting between aljabar types and mint types.
* The `serde` feature (off by default) adds serialization/deserialization support from the [serde](https://crates.io/crates/serde) crate.
* The `rand` feature (off by default) allows you to create random points, vectors, and matrices by sampling from a random number source.
* The `swizzle` feature (off by default) enables [swizzle](https://en.wikipedia.org/wiki/Swizzling_(computer_graphics)) functions for vectors.

## Contributions

Pull request of any nature are welcome, especially regaring performance improvements.
Although a aljabar is generic with respect to dimensionality, algorithms specialized 
for certain dimensions are straightforward to add and are intended to replace the 
generic algorithms along the most common code paths in the future.

## Support 

Contact the author at `map@maplant.com` or [file an issue on github.](https://github.com/maplant/aljabar/issues/new/choose)

