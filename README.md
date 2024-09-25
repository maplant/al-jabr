# al-jabr 

[![Documentation](https://docs.rs/al-jabr/badge.svg?style=for-the-badge)](https://docs.rs/al-jabr)
[![Version](https://img.shields.io/crates/v/al-jabr.svg?style=for-the-badge)](https://crates.io/crates/al-jabr)
[![Downloads](https://img.shields.io/crates/d/al-jabr.svg?style=for-the-badge)](https://crates.io/crates/al-jabr)

An n-dimensional linear algebra and mathematics library for computer
graphics and other applications, designed to be roughly compatible with
[cgmath](https://github.com/rustgd/cgmath).

The library provides:

* small vectors: `Vector2`, `Vector3`, and `Vector4` 
* points: `Point2`, `Point3`, and `Point4` 
* large column vectors: `ColumnVector<T, const N: usize>`
* matrices: `Matrix2`, `Matrix3`, `Matrix4` and `Matrix<T, const N: usize, const M: usize>`
* a quaternion type: `Quaternion`
* orthonormal (rotation) matrices: `Orthonormal`

`al-jabr` supports Vectors and Matrices of any size and will provide 
implementations for any mathematic operations that are supported by their
scalars. Additionally, `al-jabr` can leverage Rust's type system to ensure that
operations are only applied to values that are the correct size. `al-jabr` can
do this while remaining no-std compatible. 

For more information and a guide on getting started, check out the [documentation](https://docs.rs/al-jabr/).

## Cargo Features

* The `mint` feature (off by default) adds a dependency to the [mint](https://crates.io/crates/mint) crate and provides support for converting between al-jabr types and mint types.
* The `serde` feature (off by default) adds serialization/deserialization support from the [serde](https://crates.io/crates/serde) crate.
* The `rand` feature (off by default) allows you to create random points, vectors, and matrices by sampling from a random number source.
* The `swizzle` feature (off by default) enables [swizzle](https://en.wikipedia.org/wiki/Swizzling_(computer_graphics)) functions for vectors.

## Contributions

Pull request of any nature are welcome. 

## Support 

Contact the author at `maplant@protonmail.com` or [file an issue on github.](https://github.com/maplant/al-jabr/issues/new/choose)

