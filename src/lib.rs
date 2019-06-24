//! The super generic super experimental linear algebra library.
//!
//! This library serves the dual purpose of being an experimental API for
//! future rust linear algebra libraries as well as a test of rustc's strength
//! in compiling a number of in development features, such as const generics
//! and specialization.
//!
//! It is not the specific goal of this project to be useful in any sense, but
//! hopefully it will end up being roughly compatible with cgmath. On some
//! platforms at least. Which leads me into my next point:
//!
//! Aljabar is not very safe. In the attempt to make things as generic and
//! minimalist in implementation as possible, a lot of unsafe blocks are used.
//! When it is possible to specialize and make more safe implementations, that
//! is done instead.
//!
//! Almost all of the unsafe code here could be made safe if I added one or two
//! sensible trait bounds to their functions. I should probably do that.
//!
//! The performance of Aljabar is currently probably pretty bad. I have yet to
//! test it, but let's just say I haven't gotten very far on the matrix
//! multiplication page on wikipedia.
//!

#![feature(const_generics)]
#![feature(trivial_bounds)]
#![feature(specialization)]
// #![feature(maybe_uninit)]

use std::{
    fmt,
    mem,
    ops::{
        Add,
        AddAssign,
        Sub,
        SubAssign,
        Deref,
        DerefMut,
        Div,
        DivAssign,
        Mul,
        MulAssign,
        Neg,
    },
};

/// Defines the additive identity for `Self`.
pub trait Zero {
    /// Returns the additive identity of `Self`.
    fn zero() -> Self;

    /// Returns true if the value is the additive identity.
    fn is_zero(&self) -> bool;
}

macro_rules! impl_zero {
    (
        $type:ty
    ) => {
        impl Zero for $type {
            fn zero() -> Self  {
                0
            }

            fn is_zero(&self) -> bool {
                *self == 0
            }
        }
    };
}

macro_rules! impl_zero_fp {
    (
        $type:ty
    ) => {
        impl Zero for $type {
            fn zero() -> Self  {
                0.0
            }

            fn is_zero(&self) -> bool {
                *self == 0.0
            }
        }
    };
}

impl Zero for bool {
    fn zero() -> Self {
        false
    }

    fn is_zero(&self) -> bool {
        !self
    }
}

impl_zero_fp!{ f32 }
impl_zero_fp!{ f64 }
impl_zero!{ i8 }
impl_zero!{ i16 }
impl_zero!{ i32 }
impl_zero!{ i64 }
impl_zero!{ i128 }
impl_zero!{ isize }
impl_zero!{ u8 }
impl_zero!{ u16 }
impl_zero!{ u32 }
impl_zero!{ u64 }
impl_zero!{ u128 }
impl_zero!{ usize }

/// Defines the multiplicative identity element for `Self`.
pub trait One {
    /// Returns the multiplicative identity for `Self`.
    fn one() -> Self;
}

macro_rules! impl_one {
    (
        $type:ty
    ) => {
        impl One for $type {
            fn one() -> Self  {
                1
            }
        }
    };
}

macro_rules! impl_one_fp {
    (
        $type:ty
    ) => {
        impl One for $type {
            fn one() -> Self  {
                1.0
            }
        }
    };
}

impl One for bool {
    fn one() -> Self {
        true
    }
}

impl_one_fp!{ f32 }
impl_one_fp!{ f64 }
impl_one!{ i8 }
impl_one!{ i16 }
impl_one!{ i32 }
impl_one!{ i64 }
impl_one!{ i128 }
impl_one!{ isize }
impl_one!{ u8 }
impl_one!{ u16 }
impl_one!{ u32 }
impl_one!{ u64 }
impl_one!{ u128 }
impl_one!{ usize }

/// Types that have an exact square root.
pub trait Real {
    fn sqrt(self) -> Self;
}

impl Real for f32 {
    fn sqrt(self) -> Self { self.sqrt() }
}

impl Real for f64 {
    fn sqrt(self) -> Self { self.sqrt() }
}

/// N-element vector.
pub struct Vector<T, const N: usize>([T; N]);

impl<T, const N: usize> Clone for Vector<T, {N}>
where
    T: Clone
{
    fn clone(&self) -> Self {
        Vector::<T, {N}>(self.0.clone())
    }
}

impl<T, const N: usize> Copy for Vector<T, {N}>
where
    T: Copy
{}

/// 1-element vector.
pub type Vector1<T> = Vector<T, 1>;

pub fn vec1<T>(x: T) -> Vector1<T> {
    Vector1::<T>::from([ x ])
}

/// 2-element vector.
pub type Vector2<T> = Vector<T, 2>;

pub fn vec2<T>(x: T, y: T) -> Vector2<T> {
    Vector2::<T>::from([ x, y ])
}

/// 3-element vector.
pub type Vector3<T> = Vector<T, 3>;

pub fn vec3<T>(x: T, y: T, z: T) -> Vector3<T> {
    Vector3::<T>::from([ x, y, z ])
}

/// 4-element vector.
pub type Vector4<T> = Vector<T, 4>;

pub fn vec4<T>(x: T, y: T, z: T, w: T) -> Vector4<T> {
    Vector4::<T>::from([ x, y, z, w ])
}

/// 5-element vector. You don't need this, a fact that calls into question the
/// entire purpose of this library.
pub type Vector5<T> = Vector<T, 5>;

/// 6-element vector. You definitely don't need this. What kind of bespoke math
/// are you even doing?
pub type Vector6<T> = Vector<T, 6>;

/// Stop.
pub type Vector7<T> = Vector<T, 7>;

impl<T, const N: usize> From<[T; N]> for Vector<T, {N}> {
    fn from(array: [T; N]) -> Self {
        Vector::<T, {N}>(array)
    }
}

impl<T, const N: usize> Into<[T; {N}]> for Vector<T, {N}> {
    fn into(self) -> [T; {N}] {
        self.0
    }
}

impl<T, const N: usize> fmt::Debug for Vector<T, {N}>
where
    T: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match N {
            0 => unimplemented!(),
            1 => write!(f, "Vector {{ x: {:?} }}", self.0[0]),
            2 => write!(f, "Vector {{ x: {:?}, y: {:?} }}", self.0[0], self.0[1]),
            3 => write!(f, "Vector {{ x: {:?}, y: {:?}, z: {:?} }}", self.0[0], self.0[1], self.0[2]),
            4 => write!(f, "Vector {{ x: {:?}, y: {:?}, z: {:?}, w: {:?} }}", self.0[0], self.0[1], self.0[2], self.0[3]),
            _ => write!(f, "Vector {{ x: {:?}, y: {:?}, z: {:?}, w: {:?}, [..]: {:?} }}", self.0[0], self.0[1], self.0[2], self.0[3], &self.0[4..]),
        }
    }
}

impl<T, const N: usize> Deref for Vector<T, {N}> {
    type Target = [T; {N}];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for Vector<T, {N}> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const N: usize> Zero for Vector<T, {N}>
where
    T: Zero,
{
    fn zero() -> Self {
        let mut out: [T; {N}] = unsafe { mem::uninitialized() };
        for i in 0..N {
            mem::forget(
                mem::replace(
                    &mut out[i],
                    <T as Zero>::zero()
                )
            );
        }
        Vector::<T, {N}>(out)
    }

    fn is_zero(&self) -> bool {
        for i in 0..N {
            if !self.0[i].is_zero() {
                return false;
            }
        }
        true
    }
}

impl<A, B, RHS, const N: usize> PartialEq<RHS> for Vector<A, {N}>
where
    RHS: Deref<Target = [B; {N}]>,
    A: PartialEq<B>,
{
    fn eq(&self, other: &RHS) -> bool {
        for (a, b) in self.0.iter().zip(other.deref().iter()) {
            if !a.eq(b) {
                return false;
            }
        }
        true
    }
}

impl<T, const N: usize> Eq for Vector<T, {N}>
where
    T: Eq,
{}

impl<A, B, const N: usize> Add<Vector<B, {N}>> for Vector<A, {N}>
where
    A: Add<B>,
{
    type Output = Vector<<A as Add<B>>::Output, {N}>;

    fn add(mut self, rhs: Vector<B, {N}>) -> Self::Output {
        let mut out: [<A as Add<B>>::Output; {N}] = unsafe{ mem::uninitialized() };
        let mut rhs: [B; {N}] = rhs.into();
        for i in 0..N {
            out[i] =
                mem::replace(&mut self.0[i], unsafe { mem::uninitialized() }) +
                mem::replace(&mut rhs[i], unsafe { mem::uninitialized() });
        }
        mem::forget(self);
        mem::forget(rhs);
        Vector::<<A as Add<B>>::Output, {N}>(out)
    }
}

impl<A, B, const N: usize> AddAssign<Vector<B, {N}>> for Vector<A, {N}>
where
    A: AddAssign<B>,
{
    fn add_assign(&mut self, rhs: Vector<B, {N}>) {
        let mut rhs: [B; {N}] = rhs.into();
        for i in 0..N {
            self.0[i] += mem::replace(&mut rhs[i], unsafe { mem::uninitialized() });
        }
        mem::forget(rhs);
    }
}

impl<A, B, const N: usize> Sub<Vector<B, {N}>> for Vector<A, {N}>
where
    A: Sub<B>,
{
    type Output = Vector<<A as Sub<B>>::Output, {N}>;

    fn sub(mut self, rhs: Vector<B, {N}>) -> Self::Output {
        let mut out: [<A as Sub<B>>::Output; {N}] = unsafe { mem::uninitialized() };
        let mut rhs: [B; {N}] = rhs.into();
        for i in 0..N {
            out[i] =
                mem::replace(&mut self.0[i], unsafe { mem::uninitialized() }) -
                mem::replace(&mut rhs[i], unsafe { mem::uninitialized() });
        }
        mem::forget(self);
        mem::forget(rhs);
        Vector::<<A as Sub<B>>::Output, {N}>(out)
    }
}

impl<A, B, const N: usize> SubAssign<Vector<B, {N}>> for Vector<A, {N}>
where
    A: SubAssign<B>,
{
    fn sub_assign(&mut self, rhs: Vector<B, {N}>) {
        let mut rhs: [B; {N}] = rhs.into();
        for i in 0..N {
            self.0[i] -= mem::replace(&mut rhs[i], unsafe { mem::uninitialized() });
        }
        mem::forget(rhs);
    }
}

impl<T, const N: usize> Neg for Vector<T, {N}>
where
    T: Neg,
{
    type Output = Vector<<T as Neg>::Output, {N}>;

    fn neg(mut self) -> Self::Output {
        let mut vec: [<T as Neg>::Output; {N}] = unsafe { mem::uninitialized() };
        for i in 0..N {
            vec[i] = mem::replace(&mut self.0[i], unsafe { mem::uninitialized() }).neg();
        }
        mem::forget(self);
        Vector::<<T as Neg>::Output, {N}>(vec)
    }
}

/// Scalar multiply
impl<A, B, const N: usize> Mul<B> for Vector<A, {N}>
where
    A: Mul<B>,
    B: Clone,
{
    type Output = Vector<<A as Mul<B>>::Output, {N}>;

    fn mul(mut self, scalar: B) -> Self::Output {
        let mut out: [<A as Mul<B>>::Output; {N}] = unsafe { mem::uninitialized() };
        for i in 0..N {
            out[i] = mem::replace(&mut self.0[i], unsafe { mem::uninitialized() }) *
                scalar.clone();
                
        }
        mem::forget(self);
        Vector::<<A as Mul<B>>::Output, {N}>(out)
    }
}

/// Scalar multiply assign
impl<A, B, const N: usize> MulAssign<B> for Vector<A, {N}>
where
    A: MulAssign<B>,
    B: Clone,
{
    fn mul_assign(&mut self, scalar: B) {
        for i in 0..N {
            self.0[i] *= scalar.clone();
        }
    }
}

/// Scalar divide
impl<A, B, const N: usize> Div<B> for Vector<A, {N}>
where
    A: Div<B>,
    B: Clone,
{
    type Output = Vector<<A as Div<B>>::Output, {N}>;

    fn div(mut self, scalar: B) -> Self::Output {
        let mut out: [<A as Div<B>>::Output; {N}] = unsafe { mem::uninitialized() };
        for i in 0..N {
            out[i] = mem::replace(&mut self.0[i], unsafe { mem::uninitialized() }) /
                scalar.clone();                
        }
        mem::forget(self);
        Vector::<<A as Div<B>>::Output, {N}>(out)
    }
}

/// Scalar divide assign 
impl<A, B, const N: usize> DivAssign<B> for Vector<A, {N}>
where
    A: DivAssign<B>,
    B: Clone,
{
    fn div_assign(&mut self, scalar: B) {
        for i in 0..N {
            self.0[i] /= scalar.clone();
        }
    }
}

/// Vectors that can be added together and multiplied by scalars form a
/// VectorSpace.
pub trait VectorSpace
where
    Self: Sized + Zero,
    Self: Add<Self, Output = Self>,
    Self: Sub<Self, Output = Self>,
    Self: Mul<<Self as VectorSpace>::Scalar, Output = Self>,
    Self: Div<<Self as VectorSpace>::Scalar, Output = Self>,
{
    // I only need Div, but I felt like I had to add them all...
    type Scalar: Add<Self::Scalar, Output = Self::Scalar> +
        Sub<Self::Scalar, Output = Self::Scalar> +
        Mul<Self::Scalar, Output = Self::Scalar> +
        Div<Self::Scalar, Output = Self::Scalar>;
    
    fn lerp(self, other: Self, amount: Self::Scalar) -> Self;
}

impl<T, const N: usize> VectorSpace for Vector<T, {N}>
where
    T: Clone + Zero,
    T: Add<T, Output = T>,
    T: Sub<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Div<T, Output = T>,
{
    type Scalar = T;

    fn lerp(self, other: Self, amount: Self::Scalar) -> Self {
        self.clone() + ((other - self) * amount)
    }
}

/// A type with a distance function between two values.
pub trait MetricSpace: Sized {
    type Metric;

    /// Returns the distance squared between the two values.
    fn distance2(self, other: Self) -> Self::Metric;
}

/// A metric spaced where the metric is a real number.
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
    <T as MetricSpace>::Metric: Real
{}

impl<T, const N: usize> MetricSpace for Vector<T, {N}>
where
    Self: InnerSpace,
{
    type Metric = <Self as VectorSpace>::Scalar;

    fn distance2(self, other: Self) -> Self::Metric {
        (other - self).magnitude2()
    }
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
}

/// Defines an InnerSpace where the Scalar is a real number. Automatically
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
        Self::Scalar: One
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
    <T as VectorSpace>::Scalar: Real
{}

impl<T, const N: usize> InnerSpace for Vector<T, {N}>
where
    T: Clone + Zero,
    T: Add<T, Output = T>,
    T: Sub<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Div<T, Output = T>,
    // TODO: Remove this add assign bound. This is purely for ease of 
    // implementation.
    T: AddAssign<T>,
    Self: Clone, 
{
    fn dot(mut self, mut rhs: Self) -> T {
        let mut sum = <T as Zero>::zero();
        for i in 0..N {
            sum +=
                mem::replace(&mut self.0[i], unsafe { mem::uninitialized() }) *
                mem::replace(&mut rhs.0[i], unsafe { mem::uninitialized() });
        }
        mem::forget(self);
        mem::forget(rhs);
        sum
    }
}

/// An `N`-by-`M` Column Major matrix.
pub struct Matrix<T, const N: usize, const M: usize>([Vector<T, {N}>; {M}]);

/// A 1-by-1 square matrix.
pub type Mat1x1<T> = Matrix<T, 1, 1>;

/// A 2-by-2 square matrix.
pub type Mat2x2<T> = Matrix<T, 2, 2>;

/// A 3-by-3 square matrix.
pub type Mat3x3<T> = Matrix<T, 3, 3>;

/// A 4-by-4 square matrix.
pub type Mat4x4<T> = Matrix<T, 4, 4>;

impl<T, const N: usize, const M: usize> From<[Vector<T, {N}>; {M}]> for Matrix<T, {N}, {M}> {
    fn from(array: [Vector<T, {N}>; {M}]) -> Self {
        Matrix::<T, {N}, {M}>(array)
    }
}

/// Returns, uh, a 1-by-1 square matrix.
///
/// I mean, I use it for testing, so I think it's kind of cool.
pub fn mat1x1<T>(
    x00: T,
) -> Mat1x1<T> {
    Matrix::<T, 1, 1>([ Vector::<T, 1>([ x00 ]) ])
}

/// Returns a 2-by-2 square matrix. Although matrices are stored column wise,
/// the order of arguments is row by row, as a matrix would be typically
/// displayed.
pub fn mat2x2<T>(
    x00: T, x01: T,
    x10: T, x11: T,
) -> Mat2x2<T> {
    Matrix::<T, 2, 2>(
        [ Vector::<T, 2>([ x00, x10 ]),
          Vector::<T, 2>([ x01, x11 ]),  ]
    )
}

/// Returns a 3-by-3 square matrix.
pub fn mat3x3<T>(
    x00: T, x01: T, x02: T,
    x10: T, x11: T, x12: T,
    x20: T, x21: T, x22: T,
) -> Mat3x3<T> {
    Matrix::<T, 3, 3>(
        [ Vector::<T, 3>([ x00, x10, x20, ]),
          Vector::<T, 3>([ x01, x11, x21, ]),  
          Vector::<T, 3>([ x02, x12, x22, ]),  ]
    )
}

/// Returns a 4-by-4 square matrix.
pub fn mat4x4<T>(
    x00: T, x01: T, x02: T, x03: T,
    x10: T, x11: T, x12: T, x13: T,
    x20: T, x21: T, x22: T, x23: T,
    x30: T, x31: T, x32: T, x33: T,
) -> Mat4x4<T> {
    Matrix::<T, 4, 4>(
        [ Vector::<T, 4>([ x00, x10, x20, x30 ]),
          Vector::<T, 4>([ x01, x11, x21, x31 ]),  
          Vector::<T, 4>([ x02, x12, x22, x32 ]),
          Vector::<T, 4>([ x03, x13, x23, x33 ]) ]
    )
}

impl<T, const N: usize, const M: usize> Clone for Matrix<T, {N}, {M}>
where
    T: Clone
{
    fn clone(&self) -> Self {
        Matrix::<T, {N}, {M}>(self.0.clone())
    }
}

impl<T, const N: usize, const M: usize> Copy for Matrix<T, {N}, {M}>
where
    T: Copy
{}

impl<T, const N: usize, const M: usize> Deref for Matrix<T, {N}, {M}> {
    type Target = [Vector<T, {N}>; {M}];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize, const M: usize> DerefMut for Matrix<T, {N}, {M}> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<A, B, RHS, const N: usize, const M: usize> PartialEq<RHS> for Matrix<A, {N}, {M}>
where
    RHS: Deref<Target = [Vector<B, {N}>; {M}]>,
    A: PartialEq<B>,
{
    fn eq(&self, other: &RHS) -> bool {
        for (a, b) in self.0.iter().zip(other.deref().iter()) {
            if !a.eq(b) {
                return false;
            }
        }
        true
    }
}

/// I'm not quite sure how to format the debug output for a matrix. 
impl<T, const N: usize, const M: usize> fmt::Debug for Matrix<T, {N}, {M}>
where
    T: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix [ ")?;
        for i in 0..N {
            write!(f, "[ ")?;
            for j in 0..M {
                write!(f, "{:?} ", self.0[j].0[i])?;
            }
            write!(f, "] ")?;
        }
        write!(f, "]")
    }
}

/// Element-wise addition of two equal sized matrices.
impl<A, B, const N: usize, const M: usize> Add<Matrix<B, {N}, {M}>> for Matrix<A, {N}, {M}>
where
    A: Add<B>,
{
    type Output = Matrix<<A as Add<B>>::Output, {N}, {M}>;

    fn add(mut self, mut rhs: Matrix<B, {N}, {M}>) -> Self::Output {
        let mut mat: [Vector<<A as Add<B>>::Output, {N}>; {M}] = unsafe { mem::uninitialized() };
        for i in 0..M {
            for j in 0..N {
                mem::forget(
                    mem::replace(
                        &mut mat[i].0[j],
                        mem::replace(&mut self.0[i].0[j], unsafe { mem::uninitialized() }) +
                            mem::replace(&mut rhs.0[i].0[j], unsafe { mem::uninitialized() })
                    )
                )
            }
        }
        mem::forget(self);
        mem::forget(rhs);
        Matrix::<<A as Add<B>>::Output, {N}, {M}>(mat)
    }
}

impl<A, B, const N: usize, const M: usize> AddAssign<Matrix<B, {N}, {M}>> for Matrix<A, {N}, {M}>
where
    A: AddAssign<B>,
{
    fn add_assign(&mut self, mut rhs: Matrix<B, {N}, {M}>) {
        for i in 0..M {
            for j in 0..N {
                self.0[i].0[j] += mem::replace(&mut rhs.0[i].0[j], unsafe { mem::uninitialized() });
            }
        }
        mem::forget(rhs);
    }
}
    
/// Element-wise subtraction of two equal sized matrices.
impl<A, B, const N: usize, const M: usize> Sub<Matrix<B, {N}, {M}>> for Matrix<A, {N}, {M}>
where
    A: Sub<B>,
{
    type Output = Matrix<<A as Sub<B>>::Output, {N}, {M}>;

    fn sub(mut self, mut rhs: Matrix<B, {N}, {M}>) -> Self::Output {
        let mut mat: [Vector<<A as Sub<B>>::Output, {N}>; {M}] = unsafe { mem::uninitialized() };
        for i in 0..M {
            for j in 0..N {
                mem::forget(
                    mem::replace(
                        &mut mat[i].0[j],
                        mem::replace(&mut self.0[i].0[j], unsafe { mem::uninitialized() }) -
                            mem::replace(&mut rhs.0[i].0[j], unsafe { mem::uninitialized() })
                    )
                )
            }
        }
        mem::forget(self);
        mem::forget(rhs);
        Matrix::<<A as Sub<B>>::Output, {N}, {M}>(mat)
    }
}

impl<A, B, const N: usize, const M: usize> SubAssign<Matrix<B, {N}, {M}>> for Matrix<A, {N}, {M}>
where
    A: SubAssign<B>,
{
    fn sub_assign(&mut self, mut rhs: Matrix<B, {N}, {M}>) {
        for i in 0..M {
            for j in 0..N {
                self.0[i].0[j] -= mem::replace(&mut rhs.0[i].0[j], unsafe { mem::uninitialized() });
            }
        }
        mem::forget(rhs);
    }
}

impl<T, const N: usize, const M: usize> Neg for Matrix<T, {N}, {M}>
where
    T: Neg
{
    type Output = Matrix<<T as Neg>::Output, {N}, {M}>;

    fn neg(mut self) -> Self::Output {
        let mut mat: [Vector<<T as Neg>::Output, {N}>; {M}] = unsafe { mem::uninitialized() };
        for i in 0..M {
            for j in 0..N {
                mem::forget(
                    mem::replace(
                        &mut mat[i].0[j],
                        mem::replace(&mut self.0[i].0[j], unsafe { mem::uninitialized() }).neg()
                    )
                )
            }
        }
        mem::forget(self);
        Matrix::<<T as Neg>::Output, {N}, {M}>(mat)
    }
}

impl<T, const N: usize, const M: usize, const P: usize> Mul<Matrix<T, {M}, {P}>> for Matrix<T, {N}, {M}>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Clone,
    Vector<T, {M}>: InnerSpace,
{
    type Output = Matrix<<Vector<T, {M}> as VectorSpace>::Scalar, {N}, {P}>;

    fn mul(self, rhs: Matrix<T, {M}, {P}>) -> Self::Output {
        // It might not seem that Rust's type system is helping me at all here,
        // but that's absolutely not true. I got the arrays iterations wrong on
        // the first try and Rust was nice enough to inform me of that fact.
        let mut mat: [Vector<<Vector<T, {M}> as VectorSpace>::Scalar, {N}>; {P}] = unsafe { mem::uninitialized() };
        // There's a lesson here... about how these index variables should be
        // named better.
        for i in 0..P {
            let mut column: [<Vector<T, {M}> as VectorSpace>::Scalar; {N}] = unsafe { mem::uninitialized() };
            for j in 0..N {
                // Fetch the current row
                let mut row: [T; {M}] = unsafe { mem::uninitialized() };
                for k in 0..M {
                    mem::forget(mem::replace(&mut row[k], self.0[k].0[j].clone()));
                }
                let row = Vector::<T, {M}>::from(row);
                mem::forget(mem::replace(&mut column[j], row.dot(rhs.0[i].clone())))
            }
            let column = Vector::<<Vector<T, {M}> as VectorSpace>::Scalar, {N}>(column);
            mem::forget(mem::replace(&mut mat[i], column));
        }
        Matrix::<<Vector<T, {M}> as VectorSpace>::Scalar, {N}, {P}>(mat)
    }
}

impl<T, const N: usize, const M: usize> Mul<Vector<T, {M}>> for Matrix<T, {N}, {M}>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Clone,
    Vector<T, {M}>: InnerSpace,
{
    type Output = Vector<<Vector<T, {M}> as VectorSpace>::Scalar, {N}>;

    fn mul(self, rhs: Vector<T, {M}>) -> Self::Output {
        let mut column: [<Vector<T, {M}> as VectorSpace>::Scalar; {N}] = unsafe { mem::uninitialized() };
        for j in 0..N {
            // Fetch the current row
            let mut row: [T; {M}] = unsafe { mem::uninitialized() };
            for k in 0..M {
                mem::forget(mem::replace(&mut row[k], self.0[k].0[j].clone()));
            }
            let row = Vector::<T, {M}>::from(row);
            mem::forget(mem::replace(&mut column[j], row.dot(rhs.clone())))
        }
        Vector::<<Vector<T, {M}> as VectorSpace>::Scalar, {N}>(column)
    }
}

/// Scalar multiply
impl<T, const N: usize, const M: usize> Mul<T> for Matrix<T, {N}, {M}>
where
    T: Mul<T, Output = T> + Clone,
{
    type Output = Matrix<T, {N}, {M}>;

    fn mul(self, scalar: T) -> Self::Output {
        let mut mat: [Vector<T, {N}>; {M}] = unsafe { mem::uninitialized() };
        for i in 0..M {
            for j in 0..N {
                mem::forget(
                    mem::replace(
                        &mut mat[i].0[j],
                        scalar.clone() * self.0[i].0[j].clone()
                    )
                );
            }
        }
        Matrix::<T, {N}, {M}>(mat)
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, {N}, {M}> {
    /// Returns the transpose of the matrix. 
    pub fn transpose(mut self) -> Matrix<T, {M}, {N}> {
        let mut mat: [Vector<T, {M}>; {N}] = unsafe { mem::uninitialized() };
        for j in 0..N {
            // Fetch the current row
            let mut row: [T; {M}] = unsafe { mem::uninitialized() };
            for k in 0..M {
                mem::forget(
                    mem::replace(
                        &mut row[k],
                        mem::replace(
                            &mut self.0[k].0[j],
                            unsafe { mem::uninitialized() }
                        )
                    )
                );
            }
            let row = Vector::<T, {M}>::from(row);
            mem::forget(mem::replace(&mut mat[j], row));
        }
        Matrix::<T, {M}, {N}>(mat)
    }
}

/// Defines a matrix with an equal number of elements in either dimension.
///
/// Square matrices can be added, subtracted, and multiplied indiscriminately
/// together. This is a type constraint; only Matrices that are square are able
/// to be multiplied by matrices of the same size.
///
/// I believe that SquareMatrix should not have parameters, but associated types
/// and constants do not play well with const generics.
trait SquareMatrix<Scalar, const N: usize>: Sized
where
    Scalar: Clone,
    Self: Add<Self>,
    Self: Sub<Self>,
    Self: Mul<Self>,
    Self: Mul<Vector<Scalar, {N}>, Output = Vector<Scalar, {N}>>,
{
    /// Attempt to invert the matrix.
    fn invert(self) -> Option<Self>;

    /// Return the diagonal of the matrix.
    fn diagonal(&self) -> Vector<Scalar, {N}>;
}

impl<Scalar, const N: usize> SquareMatrix<Scalar, {N}> for Matrix<Scalar, {N}, {N}>
where
    Scalar: Clone,
    Self: Add<Self>,
    Self: Sub<Self>,
    Self: Mul<Self>,
    Self: Mul<Vector<Scalar, {N}>, Output = Vector<Scalar, {N}>>,
{
    fn invert(self) -> Option<Self> {
        None // Ha!
    }

    fn diagonal(&self) -> Vector<Scalar, {N}> {
        let mut diag: [Scalar; {N}] = unsafe { mem::uninitialized() };
        for i in 0..N {
            mem::forget(
                mem::replace(
                    &mut diag[i],
                    self.0[i].0[i].clone()
                )
            )
        }
        Vector::<Scalar, {N}>(diag)
    }
}
    
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_eq() {
        let a = Vector1::<u32>::from([ 0 ]);
        let b = Vector1::<u32>::from([ 1 ]);
        let c = Vector1::<u32>::from([ 0 ]);
        let d = [ 0u32 ];
        assert_ne!(a, b);
        assert_eq!(a, c);
        assert_eq!(a, &d); // No blanket impl on T for deref... why? infinite loops?
    }

    #[test]
    fn test_vec_addition() {
        let a = Vector1::<u32>::from([ 0 ]);
        let b = Vector1::<u32>::from([ 1 ]);
        let c = Vector1::<u32>::from([ 2 ]);
        assert_eq!(a + b, b);
        assert_eq!(b + b, c);
        // We shouldn't need to have to test more dimensions, but we shall test
        // one more.
        let a = Vector2::<u32>::from([ 0, 1 ]);
        let b = Vector2::<u32>::from([ 1, 2 ]);
        let c = Vector2::<u32>::from([ 1, 3 ]);
        let d = Vector2::<u32>::from([ 2, 5 ]);
        assert_eq!(a + b, c);
        assert_eq!(b + c, d);        
    }

    #[test]
    fn test_vec_distance() {
        let a = Vector1::<f32>::from([ 0.0 ]);
        let b = Vector1::<f32>::from([ 1.0 ]);
        assert_eq!(a.distance2(b), 1.0);
        let a = Vector1::<f32>::from([ 0.0 ]);
        let b = Vector1::<f32>::from([ 2.0 ]);
        assert_eq!(a.distance2(b), 4.0);
        assert_eq!(a.distance(b), 2.0);
        let a = Vector2::<f32>::from([ 0.0, 0.0 ]);
        let b = Vector2::<f32>::from([ 1.0, 1.0 ]);
        assert_eq!(a.distance2(b), 2.0);
    }

    #[test]
    fn test_vec_normalize() {
        let a = vec1(5.0);
        assert_eq!(a.clone().magnitude(), 5.0);
        let a_norm = a.normalize();
        assert_eq!(a_norm, vec1(1.0));
    }

    #[test]
    fn test_matrix_mult() {
        let a = Matrix::<f32, 2, 2>::from( [ vec2( 0.0, 0.0 ),
                                             vec2( 1.0, 0.0 ) ] );
        let b = Matrix::<f32, 2, 2>::from( [ vec2( 0.0, 1.0 ),
                                             vec2( 0.0, 0.0 ) ] );
        assert_eq!(a * b, mat2x2( 1.0, 0.0,
                                  0.0, 0.0 ));
        assert_eq!(b * a, mat2x2( 0.0, 0.0,
                                  0.0, 1.0 ));
        let a = mat1x1( mat1x1( 1u32 ) );
        let b = mat1x1( mat1x1( 10u32 ) );
        let c = mat1x1( mat1x1( 11u32 ) );
        assert_eq!(a + b, c);
        // Basic example:
        let a: Matrix::<usize, 1, 1> = mat1x1( 1 );
        let b: Matrix::<usize, 1, 1> = mat1x1( 2 );
        let c: Matrix::<usize, 1, 1> = mat1x1( 2 );
        assert_eq!(a * b, c);
        // Removing the type signature here caused the compiler to crash.
        // Since then I've been wary.
        let a = Matrix::<f32, 3, 3>::from( [ vec3( 1.0, 0.0, 0.0 ),
                                             vec3( 0.0, 1.0, 0.0 ),
                                             vec3( 0.0, 0.0, 1.0 ), ] );
        let b = a.clone();
        let c = a * b;
        assert_eq!(c, mat3x3( 1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0 ));
        // Here is another random example I found online.
        let a: Matrix::<i32, 3, 3> =
            mat3x3( 0, -3, 5,
                    6, 1, -4,
                    2, 3, -2 );
        let b: Matrix::<i32, 3, 3> =
            mat3x3( -1, 0, -3,
                     4, 5, 1,
                     2, 6, -2 );
        let c: Matrix::<i32, 3, 3> =
            mat3x3( -2, 15, -13,
                    -10, -19, -9,
                     6, 3, 1 );
        assert_eq!(
            a * b,
            c
        );
    }

    #[test]
    fn test_matrix_transpose() {
        assert_eq!(
            Matrix::<i32, 1, 2>::from( [ vec1( 1 ), vec1( 2 ) ] )
                .transpose(),
            Matrix::<i32, 2, 1>::from( [ vec2( 1, 2 ) ] )
        );
        assert_eq!(
            mat2x2( 1, 2,
                    3, 4 ).transpose(),
            mat2x2( 1, 3,
                    2, 4 )
        );
    }

    #[test]
    fn test_square_matrix() {
        let a: Matrix::<i32, 3, 3> =
            mat3x3( 5, 0, 0,
                    0, 8, 12,
                    0, 0, 16 );
        let diag: Vector::<i32, 3> =
            vec3( 5, 8, 16 );
        assert_eq!(a.diagonal(), diag);
    }
}
