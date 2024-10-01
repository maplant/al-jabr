// Copyright 2019 The Al_Jabr Developers. For a full listing of authors,
// refer to the Cargo.toml file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
//! A generic linear algebra library for computer graphics.
//!
//! `al_jabr` is roughly compatibly with [cgmath](https://github.com/rustgd/cgmath)
//! and is intended to provide a small set of lightweight linear algebra
//! operations typically useful in interactive computer graphics.
//!
//! `al_jabr` is n-dimensional, meaning that its data structures support an
//! arbitrary number of elements. If you wish to create a five-dimensional rigid
//! body simulation, `al_jabr` can help you.
//!
//! ## Getting started
//!
//! All of `al_jabr`'s types are exported in the root of the crate, so importing
//! them all is as easy as adding the following to the top of your source file:
//!
//! ```
//! use al_jabr::*;
//! ```
//!
//! After that, you can begin using `al_jabr`.
//!
//! ### Vectors
//!
//! Small (N = 1, 2, 3, 4) vectors as well as an N-dimensional [ColumnVector] are
//! provided. Unless you have a need for larger vectors, it is recommended to use
//! [Vector1], [Vector2], [Vector3] or [Vector4].
//!
//! [Add], [Sub], and [Neg] will be properly implemented for any `Vector` and
//! `ColumnVector<Scalar, N>` for any respective implementation of such operations
//! for `Scalar`. Operations are only implemented for vectors of equal sizes.
//!
//! ```
//! # use al_jabr::*;
//! let a = Vector4::new(0.0f32, 1.0, 2.0, 3.0);
//! let b = Vector4::new(1.0f32, 1.0, 1.0, 1.0);
//! assert_eq!(
//!     a + b,
//!     Vector4::new(1.0, 2.0, 3.0, 4.0),
//! );
//!
//! let d = column_vector![ 0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ];
//! let e = column_vector![ 1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ] * 0.5;
//! assert_eq!(
//!     d + e,
//!     column_vector![ 0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 ]
//! );
//! ```
//!
//! If the scalar type implements [Mul] as well, then the Vector will be an
//! [InnerSpace] and have the [dot](InnerSpace::dot) product defined for it,
//! as well as the ability to find the squared distance between two vectors
//! (implements [MetricSpace]) and  the squared magnitude of a vector. If the
//! scalar type is a real number then the  distance between two vectors and
//! the magnitude of a vector can be found in addition:
//!
//! ```rust
//! # use al_jabr::*;
//! let a = Vector2::new(1i32, 1);
//! let b = Vector2::new(5i32, 5);
//! assert_eq!(a.distance2(b), 32);       // distance method not implemented.
//! assert_eq!((b - a).magnitude2(), 32); // magnitude method not implemented.
//!
//! let a = Vector2::new(1.0f32, 1.0);
//! let b = Vector2::new(5.0f32, 5.0);
//! const close: f32 = 5.65685424949;
//! assert_eq!(a.distance(b), close);       // distance is implemented.
//! assert_eq!((b - a).magnitude(), close); // magnitude is implemented.
//!
//! // Vector normalization is also supported for floating point scalars.
//! assert_eq!(
//!     Vector3::new(0.0f32, 20.0, 0.0)
//!         .normalize(),
//!     Vector3::new(0.0f32, 1.0, 0.0)
//! );
//! ```
//!
//! ### Points
//!
//! Small (N = 1, 2, 3, 4) points in space are provided.
//!
//! Points are far less flexible and useful than vectors and are used
//! to express the purpose or meaning of a variable through its type.
//!
//! Points can be moved through space by adding or subtracting Vectors from
//! them.
//!
//! The only mathematical operator supported between two points is
//! subtraction, which results in the vector between the two points.
//!
//! Points can be freely converted to and from vectors via `from_vec`
//! and `to_vec`.
//!
//! ```rust
//! # use al_jabr::*;
//! let a = Point3::new(5, 4, 3);
//! let b = Point3::new(1, 1, 1);
//! assert_eq!(a - b, Vector3::new(4, 3, 2));
//! ```
//!
//! ### Matrices
//!
//! [Matrices](Matrix) can be created from an array of arrays of any size
//! and scalar type. Matrices are column-major and constructing a matrix from a
//! raw array reflects that. The [matrix!] macro can be used to construct a
//! matrix in row-major order:
//!
//! ```
//! # use al_jabr::*;
//! // Construct in column-major order:
//! let a = Matrix::<i32, 3, 3>::from([
//!     [  0,  6,  2 ],
//!     [ -3,  1,  3 ],
//!     [  5, -4, -2 ],
//! ]);
//!
//! // Construct in row-major order:
//! let b: Matrix::<i32, 3, 3> = matrix![
//!     [ 0, -3, 5 ],
//!     [ 6, 1, -4 ],
//!     [ 2, 3, -2 ]
//! ];
//!
//! assert_eq!(a, b);
//! ```
//!
//! All operations performed on matrices produce fixed-size outputs. For
//! example, taking the [transpose](Matrix::transpose) of a non-square matrix
//! will produce a matrix with the width and height swapped:
//!
//! ```
//! # use al_jabr::*;
//! assert_eq!(
//!     Matrix::<i32, 1, 2>::from([ [ 1 ], [ 2 ] ])
//!         .transpose(),
//!     Matrix::<i32, 2, 1>::from([ [ 1, 2 ] ])
//! );
//! ```
//!
//! As with Vectors, if the underlying scalar type supports the appropriate
//! operations, a matrix will implement element-wise [Add] and [Sub] for
//! matrices of equal size:
//!
//! ```
//! # use al_jabr::*;
//! let a = matrix!([1_u32]);
//! let b = matrix!([2_u32]);
//! let c = matrix!([3_u32]);
//! assert_eq!(a + b, c);
//! ```
//!
//! And this is true for any type that implements [Add], so therefore the
//! following is possible as well:
//!
//! ```
//! # use al_jabr::*;
//! let a = matrix!([matrix!([1_u32])]);
//! let b = matrix!([matrix!([2_u32])]);
//! let c = matrix!([matrix!([3_u32])]);
//! assert_eq!(a + b, c);
//! ```
//!
//! For a given type `T`, if `T: Clone` and `Vector<T, _>` is an [InnerSpace],
//! then multiplication is defined for `Matrix<T, N, M> * Matrix<T, M, P>`. The
//! result is a `Matrix<T, N, P>`:
//!
//! ```rust
//! # use al_jabr::*;
//! let a: Matrix::<i32, 3, 3> = matrix![
//!     [ 0, -3, 5 ],
//!     [ 6, 1, -4 ],
//!     [ 2, 3, -2 ],
//! ];
//! let b: Matrix::<i32, 3, 3> = matrix![
//!     [ -1, 0, -3 ],
//!     [  4, 5,  1 ],
//!     [  2, 6, -2 ],
//! ];
//! let c: Matrix::<i32, 3, 3> = matrix![
//!     [  -2,  15, -13 ],
//!     [ -10, -19,  -9 ],
//!     [   6,   3,   1 ],
//! ];
//! assert_eq!(
//!     a * b,
//!     c
//! );
//! ```

use core::{
    cmp::PartialOrd,
    fmt,
    hash::{Hash, Hasher},
    iter::{FromIterator, Product},
    marker::PhantomData,
    mem::{self, transmute_copy, MaybeUninit},
    ops::{
        Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub,
        SubAssign,
    },
};

#[cfg(feature = "rand")]
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

#[cfg(feature = "serde")]
use serde::{
    de::{Error, SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

mod array;
mod column_vector;
mod matrix;
mod point;
mod rotation;
pub mod row_view;
mod vector;

pub use array::*;
pub use column_vector::*;
pub use matrix::*;
pub use point::*;
pub use rotation::*;
use row_view::*;
pub use vector::*;

/// Defines the additive identity for `Self`.
pub trait Zero {
    /// Returns the additive identity of `Self`.
    fn zero() -> Self;

    /// Returns true if the value is the additive identity.
    fn is_zero(&self) -> bool;
}

macro_rules! impl_zero {
    // Default $zero to '0' if not provided.
    (
        $type:ty
    ) => {
        impl_zero! { $type, 0 }
    };
    // Main impl.
    (
        $type:ty,
        $zero:expr
    ) => {
        impl Zero for $type {
            fn zero() -> Self {
                $zero
            }

            fn is_zero(&self) -> bool {
                *self == $zero
            }
        }
    };
}

impl_zero! { bool, false }
impl_zero! { f32, 0.0 }
impl_zero! { f64, 0.0 }
impl_zero! { i8 }
impl_zero! { i16 }
impl_zero! { i32 }
impl_zero! { i64 }
impl_zero! { i128 }
impl_zero! { isize }
impl_zero! { u8 }
impl_zero! { u16 }
impl_zero! { u32 }
impl_zero! { u64 }
impl_zero! { u128 }
impl_zero! { usize }

/// Defines the multiplicative identity element for `Self`.
///
/// For Matrices, `one` is an alias for the unit matrix.
pub trait One {
    /// Returns the multiplicative identity for `Self`.
    fn one() -> Self;

    /// Returns true if the value is the multiplicative identity.
    fn is_one(&self) -> bool;
}

macro_rules! impl_one {
    // Default $one to '1' if not provided.
    (
        $type:ty
    ) => {
        impl_one! { $type, 1 }
    };
    // Main impl.
    (
        $type:ty,
        $one:expr
    ) => {
        impl One for $type {
            fn one() -> Self {
                $one
            }

            fn is_one(&self) -> bool {
                *self == $one
            }
        }
    };
}

impl_one! { bool, true }
impl_one! { f32, 1.0 }
impl_one! { f64, 1.0 }
impl_one! { i8 }
impl_one! { i16 }
impl_one! { i32 }
impl_one! { i64 }
impl_one! { i128 }
impl_one! { isize }
impl_one! { u8 }
impl_one! { u16 }
impl_one! { u32 }
impl_one! { u64 }
impl_one! { u128 }
impl_one! { usize }

/// Values that are [real numbers](https://en.wikipedia.org/wiki/Real_number#Axiomatic_approach).
pub trait Real
where
    Self: Sized,
    Self: Add<Output = Self>,
    Self: Sub<Output = Self>,
    Self: Mul<Output = Self>,
    Self: Div<Output = Self>,
    Self: Neg<Output = Self>,
    Self: PartialOrd + PartialEq,
{
    fn sqrt(self) -> Self;

    fn mul2(self) -> Self;

    fn div2(self) -> Self;

    fn abs(self) -> Self;

    /// Returns the sine of the angle.
    fn sin(self) -> Self;

    /// Returns the cosine of the angle.
    fn cos(self) -> Self;

    /// Returns the tangent of the angle.
    fn tan(self) -> Self;

    /// Returns the arcsine of the angle.
    fn asin(self) -> Self;

    /// Returns the arccos of the angle.
    fn acos(self) -> Self;

    /// Returns the four quadrant arctangent of `self` and `x` in radians.
    fn atan2(self, x: Self) -> Self;

    /// Returns the sine and the cosine of the angle.
    fn sin_cos(self) -> (Self, Self);

    /// Returns the sign of the number.
    fn signum(self) -> Self;
}

impl Real for f32 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn mul2(self) -> Self {
        2.0 * self
    }

    fn div2(self) -> Self {
        self / 2.0
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn atan2(self, x: Self) -> Self {
        self.atan2(x)
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    fn signum(self) -> Self {
        self.signum()
    }
}

impl Real for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn mul2(self) -> Self {
        2.0 * self
    }

    fn div2(self) -> Self {
        self / 2.0
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn atan2(self, x: Self) -> Self {
        self.atan2(x)
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    fn signum(self) -> Self {
        self.signum()
    }
}

/// Vectors that can be added together and multiplied by scalars form a
/// `VectorSpace`.
///
/// If a type implements [Add] and [Sub] and its scalar implements [Mul] and
/// [Div], then that vector is part of a `VectorSpace`.
pub trait VectorSpace
where
    Self: Sized + Clone + Zero,
    Self: Add<Self, Output = Self>,
    Self: Sub<Self, Output = Self>,
    Self: Mul<<Self as VectorSpace>::Scalar, Output = Self>,
    Self: Div<<Self as VectorSpace>::Scalar, Output = Self>,
{
    // I only need Div, but I felt like I had to add them all...
    type Scalar: Add<Self::Scalar, Output = Self::Scalar>
        + Sub<Self::Scalar, Output = Self::Scalar>
        + Mul<Self::Scalar, Output = Self::Scalar>
        + Div<Self::Scalar, Output = Self::Scalar>;

    /// Linear interpolate between the two vectors with a weight of `t`.
    fn lerp(self, other: Self, t: Self::Scalar) -> Self {
        self.clone() + ((other - self) * t)
    }
}

/// A type with a distance function between two values.
pub trait MetricSpace: Sized {
    type Metric;

    /// Returns the distance squared between the two values.
    fn distance2(self, other: Self) -> Self::Metric;
}

/// A [MetricSpace] where the metric is a real number.
pub trait RealMetricSpace: MetricSpace
where
    Self::Metric: Real,
{
    /// Returns the distance between the two values.
    fn distance(self, other: Self) -> Self::Metric {
        self.distance2(other).sqrt()
    }
}

impl<T> RealMetricSpace for T
where
    T: MetricSpace,
    <T as MetricSpace>::Metric: Real,
{
}

/// Vector spaces that have an inner (also known as "dot") product.
pub trait InnerSpace: VectorSpace
where
    Self: Clone,
    Self: MetricSpace<Metric = <Self as VectorSpace>::Scalar>,
{
    /// Return the inner (also known as "dot") product.
    fn dot(self, other: Self) -> Self::Scalar;

    /// Returns the squared length of the value.
    fn magnitude2(self) -> Self::Scalar {
        self.clone().dot(self)
    }

    /// Returns the [reflection](https://en.wikipedia.org/wiki/Reflection_(mathematics))
    /// of the current vector with respect to the given surface normal. The
    /// surface normal must be of length 1 for the return value to be
    /// correct. The current vector is interpreted as pointing toward the
    /// surface, and does not need to be normalized.
    fn reflect(self, surface_normal: Self) -> Self {
        let a = surface_normal.clone() * self.clone().dot(surface_normal);
        self - (a.clone() + a)
    }
}

/// Defines an [InnerSpace] where the Scalar is a real number. Automatically
/// implemented.
pub trait RealInnerSpace: InnerSpace
where
    Self: Clone,
    Self: MetricSpace<Metric = <Self as VectorSpace>::Scalar>,
    <Self as VectorSpace>::Scalar: Real,
{
    /// Returns the length of the vector.
    fn magnitude(self) -> Self::Scalar {
        self.clone().dot(self).sqrt()
    }

    /// Returns a vector with the same direction and a magnitude of `1`.
    fn normalize(self) -> Self
    where
        Self::Scalar: One,
    {
        self.normalize_to(<Self::Scalar as One>::one())
    }

    /// Returns a vector with the same direction and a given magnitude.
    fn normalize_to(self, magnitude: Self::Scalar) -> Self {
        self.clone() * (magnitude / self.magnitude())
    }

    /// Returns the
    /// [vector projection](https://en.wikipedia.org/wiki/Vector_projection)
    /// of the current inner space projected onto the supplied argument.
    fn project_on(self, other: Self) -> Self {
        other.clone() * (self.dot(other.clone()) / other.magnitude2())
    }
}

impl<T> RealInnerSpace for T
where
    T: InnerSpace,
    <T as VectorSpace>::Scalar: Real,
{
}

/// An object with a magnitude of one
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct Unit<T>(T);

impl<T> Unit<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> Deref for Unit<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Unit<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Unit<T>
where
    T: RealInnerSpace + VectorSpace,
    T::Scalar: Real + One,
{
    /// Construct a new unit object, normalizing the input in the process
    pub fn new_normalize(obj: T) -> Self {
        Unit(obj.normalize())
    }
}

impl<T> Unit<T>
where
    T: RealInnerSpace + VectorSpace + Neg<Output = T>,
    T::Scalar: Real + Zero + One + Clone,
{
    /// Perform a normalized linear interpolation between self and rhs
    pub fn nlerp(self, mut rhs: Self, amount: T::Scalar) -> Self {
        if self.0.clone().dot(rhs.0.clone()) < T::Scalar::zero() {
            rhs.0 = -rhs.0;
        }
        Self::new_normalize(self.0 * (T::Scalar::one() - amount.clone()) + rhs.0 * amount)
    }

    /// Perform a spherical linear interpolation between self and rhs
    pub fn slerp(self, mut rhs: Self, amount: T::Scalar) -> Self {
        let mut dot = self.0.clone().dot(rhs.0.clone());

        if dot.clone() < T::Scalar::zero() {
            rhs.0 = -rhs.0;
            dot = -dot;
        }

        if dot.clone() >= T::Scalar::one() {
            return self;
        }

        let theta = dot.acos();
        let scale_lhs = (theta.clone() * (T::Scalar::one() - amount.clone())).sin();
        let scale_rhs = (theta * amount).sin();

        Self::new_normalize(self.0 * scale_lhs + rhs.0 * scale_rhs)
    }
}

/// Convert a object to a unit object
pub trait IntoUnit: Sized {
    fn into_unit(self) -> Unit<Self>;
}

impl<T> IntoUnit for T
where
    T: RealInnerSpace + VectorSpace,
    T::Scalar: Real + One,
{
    fn into_unit(self) -> Unit<Self> {
        Unit::new_normalize(self)
    }
}
