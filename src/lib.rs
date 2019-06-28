//! The super generic super experimental linear algebra library.
//!
//! This library serves the dual purpose of being an experimental API for
//! future rust linear algebra libraries as well as a test of rustc's strength
//! in compiling a number of in development features, such as const generics
//! and specialization.
//!
//! It is not the specific goal of this project to be useful in any sense, but
//! hopefully it will end up being roughly compatible with cgmath. 
//!
//! The performance of Aljabar is currently probably pretty bad. I have yet to
//! test it, but let's just say I haven't gotten very far on the matrix
//! multiplication page on wikipedia.
//!

#![feature(const_generics)]
#![feature(trivial_bounds)]
#![feature(specialization)]

use std::{
    fmt,
    mem::{
        self,
        MaybeUninit,
    },
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
///
/// For Matrices, `one` is an alias for the unit matrix.
pub trait One {
    /// Returns the multiplicative identity for `Self`.
    fn one() -> Self;

    /// Returns true if the value is the multiplicative identity.
    fn is_one(&self) -> bool;
}

macro_rules! impl_one {
    (
        $type:ty
    ) => {
        impl One for $type {
            fn one() -> Self  {
                1
            }

            fn is_one(&self) -> bool {
                *self == 1
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

            fn is_one(&self) -> bool {
                *self == 1.0
            }
        }
    };
}

impl One for bool {
    fn one() -> Self {
        true
    }

    fn is_one(&self) -> bool {
        *self
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
///
/// Vectors can be constructed from arrays of any type and size. There are 
/// convenience constructor functions provided for the most common sizes.
///
/*
/// ```
/// use aljabar::*;
///
/// let a Vector::<u32, 4> = vec4( 0u32, 1, 2, 3 );
/// /*
/// assert_eq!(
///     a, 
///     Vector::<u32, 4>::from([ 0u32, 1, 2, 3 ])
/// );
/// */
/// ```
*/

#[repr(transparent)]
pub struct Vector<T, const N: usize>([T; N]);

/// A `Vector` with one fewer dimension than `N`.
///
/// Not particularly useful other than as the return value of the `trunc`
/// method.
pub type TruncatedVector<T, const N: usize> = Vector<T, {N - 1}>;

impl<T, const N: usize> Vector<T, {N}> {
    /// Drop the last component and return the vector with one fewer dimension.
    pub fn trunc(self) -> (TruncatedVector<T, {N}>, T) {
        let mut from = MaybeUninit::new(self);
        let mut head = MaybeUninit::<TruncatedVector<T, {N}>>::uninit();
        let fromp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut from) };
        let headp: *mut T = unsafe { mem::transmute(&mut head) };
        for i in 0..N {
            unsafe {
                headp.add(i).write(
                    fromp
                        .add(i)
                        .replace(MaybeUninit::uninit())
                        .assume_init()
                );
            }
        }
        (
            unsafe { head.assume_init() },
            unsafe {
                fromp
                    .add(N-1)
                    .replace(MaybeUninit::uninit())
                    .assume_init()
            }
        )
    }
}

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

/// 5-element vector. 
pub type Vector5<T> = Vector<T, 5>;

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
        let mut origin = MaybeUninit::<Vector<T, {N}>>::uninit();
        let p: *mut T = unsafe { mem::transmute(&mut origin) };

        for i in 0..N {
            unsafe {
                p.add(i).write(<T as Zero>::zero());
            }
        }

        unsafe { origin.assume_init() }
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

    fn add(self, rhs: Vector<B, {N}>) -> Self::Output {
        let mut sum = MaybeUninit::<[<A as Add<B>>::Output; {N}]>::uninit();
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let sump: *mut <A as Add<B>>::Output = unsafe { mem::transmute(&mut sum) };
        let lhsp: *mut MaybeUninit<A> = unsafe { mem::transmute(&mut lhs) };
        let rhsp: *mut MaybeUninit<B> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            unsafe {
                sump.add(i).write(
                    lhsp.add(i).replace(MaybeUninit::uninit()).assume_init() +
                        rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                );
            }
        }
        Vector::<<A as Add<B>>::Output, {N}>(unsafe { sum.assume_init() })
    }
}

impl<A, B, const N: usize> AddAssign<Vector<B, {N}>> for Vector<A, {N}>
where
    A: AddAssign<B>,
{
    fn add_assign(&mut self, rhs: Vector<B, {N}>) {
        let mut rhs = MaybeUninit::new(rhs);
        let rhsp: *mut MaybeUninit<B> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            self.0[i] += unsafe {
                rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
            };
        }
    }
}

impl<A, B, const N: usize> Sub<Vector<B, {N}>> for Vector<A, {N}>
where
    A: Sub<B>,
{
    type Output = Vector<<A as Sub<B>>::Output, {N}>;

    fn sub(self, rhs: Vector<B, {N}>) -> Self::Output {
        let mut dif = MaybeUninit::<[<A as Sub<B>>::Output; {N}]>::uninit();
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let difp: *mut <A as Sub<B>>::Output = unsafe { mem::transmute(&mut dif) };
        let lhsp: *mut MaybeUninit<A> = unsafe { mem::transmute(&mut lhs) };
        let rhsp: *mut MaybeUninit<B> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            unsafe {
                difp.add(i).write(
                    lhsp.add(i).replace(MaybeUninit::uninit()).assume_init() -
                        rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                );
            }
        }
        Vector::<<A as Sub<B>>::Output, {N}>(unsafe { dif.assume_init() })
    }
}

impl<A, B, const N: usize> SubAssign<Vector<B, {N}>> for Vector<A, {N}>
where
    A: SubAssign<B>,
{
    fn sub_assign(&mut self, rhs: Vector<B, {N}>) {
        let mut rhs = MaybeUninit::new(rhs);
        let rhsp: *mut MaybeUninit<B> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            self.0[i] -= unsafe {
                rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
            };
        }
    }
}

impl<T, const N: usize> Neg for Vector<T, {N}>
where
    T: Neg,
{
    type Output = Vector<<T as Neg>::Output, {N}>;

    fn neg(self) -> Self::Output {
        let mut from = MaybeUninit::new(self);
        let mut neg = MaybeUninit::<[<T as Neg>::Output; {N}]>::uninit();
        let fromp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut from) };
        let negp: *mut <T as Neg>::Output = unsafe { mem::transmute(&mut neg) };
        for i in 0..N {
            unsafe {
                negp.add(i).write(
                    fromp
                        .add(i)
                        .replace(MaybeUninit::uninit())
                        .assume_init()
                        .neg()
                );
            }
        }
        Vector::<<T as Neg>::Output, {N}>(unsafe { neg.assume_init() })
    }
}

/// Scalar multiply
impl<A, B, const N: usize> Mul<B> for Vector<A, {N}>
where
    A: Mul<B>,
    B: Clone,
{
    type Output = Vector<<A as Mul<B>>::Output, {N}>;

    fn mul(self, scalar: B) -> Self::Output {
        let mut from = MaybeUninit::new(self);
        let mut scaled = MaybeUninit::<[<A as Mul<B>>::Output; {N}]>::uninit();
        let fromp: *mut MaybeUninit<A> = unsafe { mem::transmute(&mut from) };
        let scaledp: *mut <A as Mul<B>>::Output =
            unsafe { mem::transmute(&mut scaled) };
        for i in 0..N {
            unsafe {
                scaledp.add(i).write(
                    fromp
                        .add(i)
                        .replace(MaybeUninit::uninit())
                        .assume_init() * scalar.clone()
                );
            }
        }
        Vector::<<A as Mul<B>>::Output, {N}>(unsafe { scaled.assume_init() })
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

    fn div(self, scalar: B) -> Self::Output {
        let mut from = MaybeUninit::new(self);
        let mut scaled = MaybeUninit::<[<A as Div<B>>::Output; {N}]>::uninit();
        let fromp: *mut MaybeUninit<A> = unsafe { mem::transmute(&mut from) };
        let scaledp: *mut <A as Div<B>>::Output =
            unsafe { mem::transmute(&mut scaled) };
        for i in 0..N {
            unsafe {
                scaledp.add(i).write(
                    fromp
                        .add(i)
                        .replace(MaybeUninit::uninit())
                        .assume_init() / scalar.clone()
                );
            }
        }
        Vector::<<A as Div<B>>::Output, {N}>(unsafe { scaled.assume_init() })
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

impl<T> Vector3<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Clone,
{
    /// Return the cross product of the two vectors.
    pub fn cross(self, rhs: Vector3<T>) -> Self {
        let [x0, y0, z0]: [T; 3] = self.into();
        let [x1, y1, z1]: [T; 3] = rhs.into();
        Vector3::from([
            (y0.clone() * z1.clone()) - (z0.clone() * y1.clone()),
            (z0 * x1.clone()) - (x0.clone() * z1),
            (x0 * y1) - (y0 * x1)
        ])
    }
}

/// Vectors that can be added together and multiplied by scalars form a
/// VectorSpace.
///
/// If a `Vector` implements `Add` and `Sub` and its scalar implements `Mul` and
/// `Div`, then that vector is part of a `VectorSpace`.
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
    fn dot(self, rhs: Self) -> T {
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let mut sum = <T as Zero>::zero();
        let lhsp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut lhs) };
        let rhsp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            sum += unsafe {
                lhsp.add(i).replace(MaybeUninit::uninit()).assume_init() *
                    rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
            };
        }
        sum
    }
}

/// An `N`-by-`M` Column Major matrix.
/// 
/// Matrices can be created from arrays of Vectors of any size and scalar type. 
/// As with Vectors there are convenience constructor functions for square matrices
/// of the most common sizes.
///
/*
/// ```
/// use aljabar::*;
///
/// let a = Matrix::<f32, 3, 3>::from( [ vec3( 1.0, 0.0, 0.0 ),
///                                      vec3( 0.0, 1.0, 0.0 ),
///                                      vec3( 0.0, 0.0, 1.0 ), ] );
/// let b: Matrix::<i32, 3, 3> =
///             mat3x3( 0, -3, 5,
///                     6, 1, -4,
///                     2, 3, -2 );
/// ```
/// 
*/
/// All operations performed on matrices produce fixed-size outputs. For example,
/// taking the `transpose` of a non-square matrix will produce a matrix with the 
/// width and height swapped: 
///
/*
/// ```
/// /*
/// use aljabar::*;
///
/// assert_eq!(
///     Matrix::<i32, 1, 2>::from( [ vec1( 1 ), vec1( 2 ) ] )
///         .transpose(),
///     Matrix::<i32, 2, 1>::from( [ vec2( 1, 2 ) ] )
/// );
/// */
/// ```
 */
#[repr(transparent)]
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

impl<T, const N: usize, const M: usize> Zero for Matrix<T, {N}, {M}>
where
    T: Zero,
// This bound is a consequence of the previous, but I'm going to preemptively
// help out the compiler a bit on this one.
    Vector<T, {N}>: Zero,
{
    fn zero() -> Self {
        let mut zero_mat = MaybeUninit::<[Vector<T, {N}>; {M}]>::uninit();
        let matp: *mut Vector<T, {N}> =
            unsafe { mem::transmute(&mut zero_mat) };
        for i in 0..M {
            unsafe { matp.add(i).write(Vector::<T, {N}>::zero()); }
        }
        Matrix::<T, {N}, {M}>(unsafe { zero_mat.assume_init() })
    }

    fn is_zero(&self) -> bool {
        for i in 0..M {
            if !self.0[i].is_zero() {
                return false;
            }
        }
        true
    }
}

/// Constructs a unit matrix.
impl<T, const N: usize> One for Matrix<T, {N}, {N}>
where
    T: Zero + One + Clone,
    Self: PartialEq<Self> + SquareMatrix<T, {N}>,
{
    fn one() -> Self {
        let mut unit_mat = MaybeUninit::<[Vector<T, {N}>; {N}]>::uninit();
        let matp: *mut Vector<T, {N}> =
            unsafe { mem::transmute(&mut unit_mat) };
        for i in 0..N {
            let mut unit_vec = MaybeUninit::<Vector<T, {N}>>::uninit();
            let vecp: *mut T = unsafe { mem::transmute(&mut unit_vec) };
            for j in 0..i {
                unsafe {
                    vecp.add(j).write(<T as Zero>::zero());
                }
            }
            unsafe { vecp.add(i).write(<T as One>::one()); }
            for j in (i+1)..N {
                unsafe {
                    vecp.add(j).write(<T as Zero>::zero());
                }
            }
            unsafe { matp.add(i).write(unit_vec.assume_init()); }
        }
        Matrix::<T, {N}, {N}>(unsafe { unit_mat.assume_init() })
    }

    fn is_one(&self) -> bool {
        self == &<Self as One>::one()
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

    fn add(self, rhs: Matrix<B, {N}, {M}>) -> Self::Output {
        let mut mat =
            MaybeUninit::<[Vector<<A as Add<B>>::Output, {N}>; {M}]>::uninit();
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let matp: *mut Vector<<A as Add<B>>::Output, {N}> =
            unsafe { mem::transmute(&mut mat) };
        let lhsp: *mut MaybeUninit<Vector<A, {N}>> =
            unsafe { mem::transmute(&mut lhs) };
        let rhsp: *mut MaybeUninit<Vector<B, {N}>> =
            unsafe { mem::transmute(&mut rhs) };
        for i in 0..M {
            unsafe {
                matp.add(i).write(
                    lhsp.add(i).replace(MaybeUninit::uninit()).assume_init() +
                        rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                );
            }
        }
        Matrix::<<A as Add<B>>::Output, {N}, {M}>(unsafe { mat.assume_init() })
    }
}

impl<A, B, const N: usize, const M: usize> AddAssign<Matrix<B, {N}, {M}>> for Matrix<A, {N}, {M}>
where
    A: AddAssign<B>,
{
    fn add_assign(&mut self, rhs: Matrix<B, {N}, {M}>) {
        let mut rhs = MaybeUninit::new(rhs);
        let rhsp: *mut MaybeUninit<Vector<B, {N}>> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..M {
            self.0[i] += unsafe {
                rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
            };
        }
    }
}
    
/// Element-wise subtraction of two equal sized matrices.
impl<A, B, const N: usize, const M: usize> Sub<Matrix<B, {N}, {M}>> for Matrix<A, {N}, {M}>
where
    A: Sub<B>,
{
    type Output = Matrix<<A as Sub<B>>::Output, {N}, {M}>;

    fn sub(self, rhs: Matrix<B, {N}, {M}>) -> Self::Output {
        let mut mat =
            MaybeUninit::<[Vector<<A as Sub<B>>::Output, {N}>; {M}]>::uninit();
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let matp: *mut Vector<<A as Sub<B>>::Output, {N}> =
            unsafe { mem::transmute(&mut mat) };
        let lhsp: *mut MaybeUninit<Vector<A, {N}>> =
            unsafe { mem::transmute(&mut lhs) };
        let rhsp: *mut MaybeUninit<Vector<B, {N}>> =
            unsafe { mem::transmute(&mut rhs) };
        for i in 0..M {
            unsafe {
                matp.add(i).write(
                    lhsp.add(i).replace(MaybeUninit::uninit()).assume_init() -
                        rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                );
            }
        }
        Matrix::<<A as Sub<B>>::Output, {N}, {M}>(unsafe { mat.assume_init() })
    }
}

impl<A, B, const N: usize, const M: usize> SubAssign<Matrix<B, {N}, {M}>> for Matrix<A, {N}, {M}>
where
    A: SubAssign<B>,
{
    fn sub_assign(&mut self, rhs: Matrix<B, {N}, {M}>) {
        let mut rhs = MaybeUninit::new(rhs);
        let rhsp: *mut MaybeUninit<Vector<B, {N}>> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..M {
            self.0[i] -= unsafe {
                rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
            };
        }
    }
}

impl<T, const N: usize, const M: usize> Neg for Matrix<T, {N}, {M}>
where
    T: Neg
{
    type Output = Matrix<<T as Neg>::Output, {N}, {M}>;

    fn neg(self) -> Self::Output {
        let mut from = MaybeUninit::new(self);
        let mut mat =
            MaybeUninit::<[Vector<<T as Neg>::Output, {N}>; {M}]>::uninit();
        let fromp: *mut MaybeUninit<Vector<T, {N}>> = unsafe { mem::transmute(&mut from) };
        let matp: *mut Vector<<T as Neg>::Output, {N}> =
            unsafe { mem::transmute(&mut mat) };
        for i in 0..M {
            unsafe {
                matp.add(i).write(
                    fromp
                        .add(i)
                        .replace(MaybeUninit::uninit())
                        .assume_init()
                        .neg()
                );
            }
        }
        Matrix::<<T as Neg>::Output, {N}, {M}>(unsafe { mat.assume_init() })
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
        let mut mat = MaybeUninit::<[Vector<<Vector<T, {M}> as VectorSpace>::Scalar, {N}>; {P}]>::uninit();
        let matp: *mut Vector<<Vector<T, {M}> as VectorSpace>::Scalar, {N}> =
            unsafe { mem::transmute(&mut mat) };
        for i in 0..P {
            let mut column = MaybeUninit::<[<Vector<T, {M}> as VectorSpace>::Scalar; {N}]>::uninit();
            let columnp: *mut <Vector<T, {M}> as VectorSpace>::Scalar =
                unsafe { mem::transmute(&mut column) };
            for j in 0..N {
                // Fetch the current row:
                let mut row = MaybeUninit::<[T; {M}]>::uninit();
                let rowp: *mut T = unsafe { mem::transmute(&mut row) };
                for k in 0..M {
                    unsafe {
                        rowp.add(k).write(self.0[k].0[j].clone());
                    }
                }
                let row = Vector::<T, {M}>::from(unsafe { row.assume_init() });
                unsafe {
                    columnp.add(j).write(row.dot(rhs.0[i].clone()));
                }
            }
            let column = Vector::<<Vector<T, {M}> as VectorSpace>::Scalar, {N}>(
                unsafe { column.assume_init() }
            );
            unsafe {
                matp.add(i).write(column);
            }
        }
        Matrix::<<Vector<T, {M}> as VectorSpace>::Scalar, {N}, {P}>(
            unsafe { mat.assume_init() }
        )
    }
}

impl<T, const N: usize, const M: usize> Mul<Vector<T, {M}>> for Matrix<T, {N}, {M}>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Clone,
    Vector<T, {M}>: InnerSpace,
{
    type Output = Vector<<Vector<T, {M}> as VectorSpace>::Scalar, {N}>;

    fn mul(self, rhs: Vector<T, {M}>) -> Self::Output {
        let mut column = MaybeUninit::<[<Vector<T, {M}> as VectorSpace>::Scalar; {N}]>::uninit();
        let columnp: *mut <Vector<T, {M}> as VectorSpace>::Scalar =
            unsafe { mem::transmute(&mut column) };
        for j in 0..N {
            // Fetch the current row:
            let mut row = MaybeUninit::<[T; {M}]>::uninit();
            let rowp: *mut T = unsafe { mem::transmute(&mut row) };
            for k in 0..M {
                unsafe {
                    rowp.add(k).write(self.0[k].0[j].clone());
                }
            }
            let row = Vector::<T, {M}>::from(unsafe { row.assume_init() });
            unsafe {
                columnp.add(j).write(row.dot(rhs.clone()));
            }
        }
        Vector::<<Vector<T, {M}> as VectorSpace>::Scalar, {N}>(
            unsafe { column.assume_init() }
        )
    }
}

/// Scalar multiply
impl<T, const N: usize, const M: usize> Mul<T> for Matrix<T, {N}, {M}>
where
    T: Mul<T, Output = T> + Clone,
{
    type Output = Matrix<T, {N}, {M}>;

    fn mul(self, scalar: T) -> Self::Output {
        let mut mat = MaybeUninit::<[Vector<T, {N}>; {M}]>::uninit();
        let matp: *mut Vector<T, {N}> = unsafe { mem::transmute(&mut mat) };
        for i in 0..M {
            unsafe {
                matp.add(i).write(self.0[i].clone() * scalar.clone());
            }
        }
        Matrix::<T, {N}, {M}>(unsafe { mat.assume_init() })
    }
}

impl<T, const N: usize, const M: usize> Matrix<T, {N}, {M}> {
    /// Returns the transpose of the matrix. 
    pub fn transpose(self) -> Matrix<T, {M}, {N}> {
        let mut from = MaybeUninit::new(self);
        let mut trans = MaybeUninit::<[Vector<T, {M}>; {N}]>::uninit();
        let fromp: *mut Vector<MaybeUninit<T>, {N}> = unsafe { mem::transmute(&mut from) };
        let transp: *mut Vector<T, {M}> = unsafe{ mem::transmute(&mut trans) };
        for j in 0..N {
            // Fetch the current row
            let mut row = MaybeUninit::<[T; {M}]>::uninit();
            let rowp: *mut T = unsafe { mem::transmute(&mut row) };
            for k in 0..M {
                unsafe {
                    let fromp: *mut MaybeUninit<T> = mem::transmute(fromp.add(k));
                    rowp.add(k).write(
                        fromp
                            .add(j)
                            .replace(MaybeUninit::uninit())
                            .assume_init()
                    );
                }
            }
            let row = Vector::<T, {M}>::from(unsafe { row.assume_init() });
            unsafe {
                transp.add(j).write(row);
            }
        }
        Matrix::<T, {M}, {N}>(unsafe { trans.assume_init() })
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
pub trait SquareMatrix<Scalar, const N: usize>: Sized
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
        unimplemented!()
    }

    fn diagonal(&self) -> Vector<Scalar, {N}> {
        let mut diag = MaybeUninit::<[Scalar; {N}]>::uninit();
        let diagp: *mut Scalar = unsafe { mem::transmute(&mut diag) };
        for i in 0..N {
            unsafe {
                diagp.add(i).write(self.0[i].0[i].clone());
            }
        }
        Vector::<Scalar, {N}>(unsafe { diag.assume_init() })
    }
}
    
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_zero() {
        let a = Vector3::<u32>::zero();
        assert_eq!(a, Vector3::<u32>::from([ 0, 0, 0 ]));
    }

    #[test]
    fn test_vec_index() {
        let a = Vector1::<u32>::from([ 0 ]);
        assert_eq!(a[0], 0);
        let mut b = Vector2::<u32>::from([ 1, 2 ]);
        b[1] += 3;
        assert_eq!(b[1], 5);
    }

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

    // This Does not compile unfortunately:
    /*
    #[test]
    fn test_vec_trunc() {
        
        let (xyz, w): (TruncatedVector<_, 4>, _) = vec4(0u32, 1, 2, 3).trunc();
    }
    */

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
        let mut c = Vector2::<u32>::from([ 1, 3 ]);
        let d = Vector2::<u32>::from([ 2, 5 ]);
        c += d;
        let e = Vector2::<u32>::from([ 3, 8 ]);
        assert_eq!(c, e);
    }

    #[test]
    fn test_vec_subtraction() {
        let mut a = Vector1::<u32>::from([ 3 ]);
        let b = Vector1::<u32>::from([ 1 ]);
        let c = Vector1::<u32>::from([ 2 ]);
        assert_eq!(a - c, b);
        a -= b;
        assert_eq!(a, c);
    }

    #[test]
    fn test_vec_negation() {
        let a = Vector4::<i32>::from([ 1, 2, 3, 4 ]);
        let b = Vector4::<i32>::from([ -1, -2, -3, -4 ]);
        assert_eq!(-a, b);
    }

    #[test]
    fn test_vec_scale() {
        let a = Vector4::<f32>::from([ 2.0, 4.0, 2.0, 4.0 ]);
        let b = Vector4::<f32>::from([ 4.0, 8.0, 4.0, 8.0 ]);
        let c = Vector4::<f32>::from([ 1.0, 2.0, 1.0, 2.0 ]);
        assert_eq!(a * 2.0, b);
        assert_eq!(a / 2.0, c);
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
    fn test_mat_identity() {
        let unit = mat4x4( 1u32, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           0, 0, 0, 1 );
        assert_eq!(
            Matrix::<u32, 4, 4>::one(),
            unit
        );
    }

    #[test]
    fn test_mat_negation() {
        let neg_unit = mat4x4( -1i32, 0, 0, 0,
                                0, -1, 0, 0,
                                0, 0, -1, 0,
                                0, 0, 0, -1 );
        assert_eq!(
            -Matrix::<i32, 4, 4>::one(),
            neg_unit
        );
    }

    #[test]
    fn test_mat_add() {
        let a = mat1x1( mat1x1( 1u32 ) );
        let b = mat1x1( mat1x1( 10u32 ) );
        let c = mat1x1( mat1x1( 11u32 ) );
        assert_eq!(a + b, c);
    }

    #[test]
    fn test_mat_scalar_mult() {
        let a = Matrix::<f32, 2, 2>::from( [ vec2( 0.0, 1.0 ),
                                             vec2( 0.0, 2.0 ) ] );
        let b = Matrix::<f32, 2, 2>::from( [ vec2( 0.0, 2.0 ),
                                             vec2( 0.0, 4.0 ) ] );
        assert_eq!(a * 2.0, b);
    }

    #[test]
    fn test_mat_mult() {
        let a = Matrix::<f32, 2, 2>::from( [ vec2( 0.0, 0.0 ),
                                             vec2( 1.0, 0.0 ) ] );
        let b = Matrix::<f32, 2, 2>::from( [ vec2( 0.0, 1.0 ),
                                             vec2( 0.0, 0.0 ) ] );
        assert_eq!(a * b, mat2x2( 1.0, 0.0,
                                  0.0, 0.0 ));
        assert_eq!(b * a, mat2x2( 0.0, 0.0,
                                  0.0, 1.0 ));
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
    fn test_mat_transpose() {
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

    #[test]
    fn test_readme_code() {
        let a = vec4( 0u32, 1, 2, 3 ); 
        assert_eq!(
	          a, 
            Vector::<u32, 4>::from([ 0u32, 1, 2, 3 ])
        );

        let b = Vector::<f32, 7>::from([ 0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ]);
        let c = Vector::<f32, 7>::from([ 1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]) * 0.5; 
        assert_eq!(
            b + c, 
            Vector::<f32, 7>::from([ 0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 ])
        );

        let a = vec2( 1i32, 1);
        let b = vec2( 5i32, 5 );
        assert_eq!(a.distance2(b), 32);       // distance method not implemented.
        assert_eq!((b - a).magnitude2(), 32); // magnitude method not implemented.

        let a = vec2( 1.0f32, 1.0 );
        let b = vec2( 5.0f32, 5.0 );
        const CLOSE: f32 = 5.65685424949;
        assert_eq!(a.distance(b), CLOSE);       // distance is implemented.
        assert_eq!((b - a).magnitude(), CLOSE); // magnitude is implemented.

        // Vector normalization is also supported for floating point scalars.
        assert_eq!(
            vec3( 0.0f32, 20.0, 0.0 )
                .normalize(),
            vec3( 0.0f32, 1.0, 0.0 )
        );

        let _a = Matrix::<f32, 3, 3>::from( [ vec3( 1.0, 0.0, 0.0 ),
                                              vec3( 0.0, 1.0, 0.0 ),
                                              vec3( 0.0, 0.0, 1.0 ), ] );
        let _b: Matrix::<i32, 3, 3> =
            mat3x3( 0, -3, 5,
                    6, 1, -4,
                    2, 3, -2 );

        assert_eq!(
            mat3x3( 1i32, 0, 0,
                    0, 2, 0,
                    0, 0, 3 )
                .diagonal(),
            vec3( 1i32, 2, 3 ) 
        );

        assert_eq!(
            mat4x4( 1i32, 0, 0, 0, 
                    0, 2, 0, 0, 
                    0, 0, 3, 0, 
                    0, 0, 0, 4 )
                .diagonal(),
            vec4( 1i32, 2, 3, 4 ) 
        );
    }
}
