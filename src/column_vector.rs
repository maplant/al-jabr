//! `N`-element vector.
use super::*;

/// `N`-element vector.
///
/// Vectors can be constructed from arrays of any type and size. There are
/// convenience constructor functions provided for the most common sizes.
///
/// ```
/// # use al_jabr::*;
/// let a: Vector::<u32, 4> = vector!( 0u32, 1, 2, 3 );
/// assert_eq!(
///     a,
///     Vector::<u32, 4>::from([ 0u32, 1, 2, 3 ])
/// );
/// ```
#[cfg_attr(
    feature = "swizzle",
    doc = r##"
# Swizzling
[Swizzling](https://en.wikipedia.org/wiki/Swizzling_(computer_graphics))
is supported for up to four elements. Swizzling is a technique for easily
rearranging and accessing elements of a vector, used commonly in graphics
shader programming. Swizzling is available on vectors whose element type
implements `Clone`.
Single-element accessors return the element itself. Multi-element accessors
return vectors of the appropriate size.
## Element names
Only the first four elements of a vector may be swizzled. If you have vectors
larger than length four and want to manipulate their elements, you must do so
manually.
Because swizzling is often used in compute graphics contexts when dealing with
colors, both 'xyzw' and 'rgba' element names are available.

| Element Index | xyzw Name | rgba Name |
|---------------|-----------|-----------|
| 0             | x         | r         |
| 1             | y         | g         |
| 2             | z         | b         |
| 3             | w         | a         |

## Restrictions
It is a runtime error to attempt to access an element beyond the bounds of a vector.
For example, `vec2(1i32, 2).z()` will panic because `z()` is only available on vectors
of length 3 or greater. Previously, this was a compilation error. However, for newer
versions of rustc this is no longer always the case.
```should_panic
# use al_jabr::*;
let z = vector!(1i32, 2).z(); // Will panic.
```
### Mixing
zle methods are not implemented for mixed xyzw/rgba methods.
```
# use al_jabr::*;
let v = vector!(1i32, 2, 3, 4);
let xy = v.xy(); // OK, only uses xyzw names.
let ba = v.ba(); // OK, only uses rgba names.
assert_eq!(xy, vector!(1i32, 2));
assert_eq!(ba, vector!(3i32, 4));
```
```compile_fail
# use al_jabr::*;
let v = vector!(1i32, 2, 3, 4);
let bad = v.xyrg(); // Compile error, mixes xyzw and rgba names.
```
## Examples
To get the first two elements of a 4-vector.
```
# use al_jabr::*;
let v = vector!(1i32, 2, 3, 4).xy();
```
To get the first and last element of a 4-vector.
```
# use al_jabr::*;
let v = vector!(1i32, 2, 3, 4).xw();
```
To reverse the order of a 3-vector.
```
# use al_jabr::*;
let v = vector!(1i32, 2, 3).zyx();
```
To select the first and third elements into the second and fourth elements,
respectively.
```
# use al_jabr::*;
let v = vector!(1i32, 2, 3, 4).xxzz();
```
"##
)]
pub type ColumnVector<T, const N: usize> = Matrix<T, N, 1>;

impl<T> ColumnVector<T, 1>
where
    T: One,
{
    /// Construct a Vector1 in the positive x direction.
    pub fn unit_x() -> Self {
        Matrix([[T::one()]])
    }
}

impl<T> ColumnVector<T, 2>
where
    T: One + Zero,
{
    /// Construct a ColumnVector2 in the positive x direction.
    pub fn unit_x() -> Self {
        Matrix([[T::one(), T::zero()]])
    }

    /// Construct a ColumnVector2 in the positive y direction.
    pub fn unit_y() -> Self {
        Matrix([[T::zero(), T::one()]])
    }
}

impl<T> ColumnVector<T, 3>
where
    T: One + Zero,
{
    /// Construct a ColumnVector3 in the positive x direction.
    pub fn unit_x() -> Self {
        Matrix([[T::one(), T::zero(), T::zero()]])
    }

    /// Construct a ColumnVector3 in the positive y direction.
    pub fn unit_y() -> Self {
        Matrix([[T::zero(), T::one(), T::zero()]])
    }

    /// Construct a ColumnVector3 in the positive z direction.
    pub fn unit_z() -> Self {
        Matrix([[T::zero(), T::zero(), T::one()]])
    }
}

impl<T> ColumnVector<T, 4>
where
    T: One + Zero,
{
    /// Construct a ColumnVector4 in the positive x direction.
    pub fn unit_x() -> Self {
        Matrix([[T::one(), T::zero(), T::zero(), T::zero()]])
    }

    /// Construct a ColumnVector4 in the positive y direction.
    pub fn unit_y() -> Self {
        Matrix([[T::zero(), T::one(), T::zero(), T::zero()]])
    }

    /// Construct a ColumnVector4 in the positive z direction.
    pub fn unit_z() -> Self {
        Matrix([[T::zero(), T::zero(), T::one(), T::zero()]])
    }

    /// Construct a ColumnVector4 in the positive w direction.
    pub fn unit_w() -> Self {
        Matrix([[T::zero(), T::zero(), T::zero(), T::one()]])
    }
}

impl<T, const N: usize> ColumnVector<T, N> {
    /// Convert the Vector into its inner array.
    pub fn into_inner(self) -> [T; N] {
        let Matrix([inner]) = self;
        inner
    }

    /// Constructs a new vector whose elements are equal to the value of the
    /// given function evaluated at the element's index.
    pub fn from_fn<Out, F>(mut f: F) -> ColumnVector<Out, N>
    where
        F: FnMut(usize) -> Out,
    {
        let mut to = MaybeUninit::<ColumnVector<Out, N>>::uninit();
        let top = &mut to as *mut MaybeUninit<Matrix<Out, N, 1>> as *mut Out;
        for i in 0..N {
            unsafe { top.add(i).write(f(i)) }
        }
        unsafe { to.assume_init() }
    }

    pub fn indexed_map<Out, F>(self, mut f: F) -> ColumnVector<Out, N>
    where
        F: FnMut(usize, T) -> Out,
    {
        let mut from = MaybeUninit::new(self);
        let mut to = MaybeUninit::<ColumnVector<Out, N>>::uninit();
        let fromp = &mut from as *mut MaybeUninit<Matrix<T, N, 1>> as *mut MaybeUninit<T>;
        let top = &mut to as *mut MaybeUninit<Matrix<Out, N, 1>> as *mut Out;
        for i in 0..N {
            unsafe {
                top.add(i).write(f(
                    i,
                    fromp.add(i).replace(MaybeUninit::uninit()).assume_init(),
                ));
            }
        }
        unsafe { to.assume_init() }
    }
}

/*
/// A `Vector` with one fewer dimension than `N`.
///
/// Not particularly useful other than as the return value of the
/// [truncate](Vector::truncate) method.
#[doc(hidden)]
pub type TruncatedVector<T, const N: usize> = Vector<T, { N - 1 }>;

/// A `Vector` with one more additional dimension than `N`.
///
/// Not particularly useful other than as the return value of the
/// [extend](Vector::extend) method.
#[doc(hidden)]
pub type ExtendedVector<T, const N: usize> = Vector<T, { N + 1 }>;
*/

impl<T, const N: usize> ColumnVector<T, N>
where
    T: Clone,
{
    /// Returns the first `M` elements of `self` in an appropriately sized
    /// `Vector`.
    ///
    /// Calling `first` with `M > N` is a compile error.
    pub fn first<const M: usize>(&self) -> ColumnVector<T, { M }> {
        if M > N {
            panic!("attempt to return {} elements from a {}-vector", M, N);
        }
        let mut head = MaybeUninit::<ColumnVector<T, { M }>>::uninit();
        let headp = &mut head as *mut MaybeUninit<Matrix<T, M, 1>> as *mut T;
        for i in 0..M {
            unsafe {
                headp.add(i).write(self.0[0][i].clone());
            }
        }
        unsafe { head.assume_init() }
    }

    /// Returns the last `M` elements of `self` in an appropriately sized
    /// `Vector`.
    ///
    /// Calling `last` with `M > N` is a compile error.
    pub fn last<const M: usize>(&self) -> ColumnVector<T, { M }> {
        if M > N {
            panic!("attempt to return {} elements from a {}-vector", M, N);
        }
        let mut tail = MaybeUninit::<ColumnVector<T, { M }>>::uninit();
        let tailp = &mut tail as *mut MaybeUninit<Matrix<T, M, 1>> as *mut T;
        for i in 0..M {
            unsafe {
                tailp.add(i + N - M).write(self.0[0][i].clone());
            }
        }
        unsafe { tail.assume_init() }
    }
}

impl<T> ColumnVector3<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Clone,
{
    /// Return the cross product of the two vectors.
    pub fn cross(self, rhs: ColumnVector3<T>) -> Self {
        let Matrix([[x0, y0, z0]]) = self;
        let Matrix([[x1, y1, z1]]) = rhs;
        Self::from([[
            (y0.clone() * z1.clone()) - (z0.clone() * y1.clone()),
            (z0 * x1.clone()) - (x0.clone() * z1),
            (x0 * y1) - (y0 * x1),
        ]])
    }
}

impl<T, const N: usize> ColumnVector<T, N>
where
    T: Clone + PartialOrd,
{
    /// Return the largest value found in the vector, along with the
    /// associated index.
    pub fn argmax(&self) -> (usize, T) {
        let mut i_max = 0;
        let mut v_max = self.0[0][0].clone();
        for i in 1..N {
            if self.0[0][i] > v_max {
                i_max = i;
                v_max = self.0[0][i].clone();
            }
        }
        (i_max, v_max)
    }

    /// Return the largest value in the vector.
    pub fn max(&self) -> T {
        let mut v_max = self.0[0][0].clone();
        for i in 1..N {
            if self.0[0][i] > v_max {
                v_max = self.0[0][i].clone();
            }
        }
        v_max
    }

    /// Return the smallest value found in the vector, along with the
    /// associated index.
    pub fn argmin(&self) -> (usize, T) {
        let mut i_min = 0;
        let mut v_min = self.0[0][0].clone();
        for i in 1..N {
            if self.0[0][i] < v_min {
                i_min = i;
                v_min = self.0[0][i].clone();
            }
        }
        (i_min, v_min)
    }

    /// Return the smallest value in the vector.
    pub fn min(&self) -> T {
        let mut v_min = self.0[0][0].clone();
        for i in 1..N {
            if self.0[0][i] < v_min {
                v_min = self.0[0][i].clone();
            }
        }
        v_min
    }
}

impl<T, const N: usize> From<[T; N]> for ColumnVector<T, N> {
    fn from(array: [T; N]) -> Self {
        Matrix([array])
    }
}

impl<T, const N: usize> From<ColumnVector<T, N>> for [T; N] {
    fn from(v: ColumnVector<T, N>) -> [T; N] {
        let Matrix([vec]) = v;
        vec
    }
}

/// 1-element vector.
pub type ColumnVector1<T> = ColumnVector<T, 1>;

/// 2-element vector.
pub type ColumnVector2<T> = ColumnVector<T, 2>;

impl<T> ColumnVector2<T> {
    /// Extend a Vector1 into a Vector2.
    pub fn from_vec1(v: ColumnVector1<T>, y: T) -> Self {
        let Matrix([[x]]) = v;
        Matrix([[x, y]])
    }
}

/// 3-element vector.
pub type ColumnVector3<T> = ColumnVector<T, 3>;

impl<T> ColumnVector3<T> {
    /// Extend a Vector1 into a Vector3.
    pub fn from_vec1(v: ColumnVector1<T>, y: T, z: T) -> Self {
        let Matrix([[x]]) = v;
        Matrix([[x, y, z]])
    }

    /// Extend a Vector2 into a Vector3.
    pub fn from_vec2(v: ColumnVector2<T>, z: T) -> Self {
        let Matrix([[x, y]]) = v;
        Matrix([[x, y, z]])
    }
}

/// 4-element vector.
pub type ColumnVector4<T> = ColumnVector<T, 4>;

impl<T> ColumnVector4<T> {
    /// Extend a Vector1 into a Vector4.
    pub fn from_vec1(v: ColumnVector1<T>, y: T, z: T, w: T) -> Self {
        let Matrix([[x]]) = v;
        Matrix([[x, y, z, w]])
    }

    /// Extend a Vector2 into a Vector4.
    pub fn from_vec2(v: ColumnVector2<T>, z: T, w: T) -> Self {
        let Matrix([[x, y]]) = v;
        Matrix([[x, y, z, w]])
    }

    /// Extend a Vector3 into a Vector4.
    pub fn from_vec3(v: ColumnVector3<T>, w: T) -> Self {
        let Matrix([[x, y, z]]) = v;
        Matrix([[x, y, z, w]])
    }
}

/// Construct a new [ColumnVector] of any size.
///
/// ```
/// # use al_jabr::*;
/// let v: ColumnVector<u32, 0> = column_vector![];
/// let v = column_vector![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let v = column_vector![true, false, false, true];
/// ```
#[macro_export]
macro_rules! column_vector {
    ( $($elem:expr),* $(,)? ) => {
        $crate::matrix![
            $([ $elem ]),*
        ]
    }
}

// @EkardNT: The cool thing about this is that Rust apparently monomorphizes
// only those functions which are actually used. This means that this impl for
// vectors of any length N is able to support vectors of length N < 4. For
// example, calling x() on a Vector2 works, but attempting to call z() will
// result in a nice compile error.
//
// @maplant: Unfortunately, I think due to a compiler change this is no longer
// the case. I sure hope it's brought back, however...
impl<T, const N: usize> ColumnVector<T, N> {
    /// Alias for `.get(0)`.
    ///
    /// # Panics
    /// When `N` = 0.
    pub fn x(&self) -> &T {
        &self.0[0][0]
    }

    pub fn x_mut(&mut self) -> &mut T {
        &mut self.0[0][0]
    }

    /// Alias for `.get(1)`.
    ///
    /// # Panics
    /// When `N` < 2.
    pub fn y(&self) -> &T {
        &self.0[0][1]
    }

    pub fn y_mut(&mut self) -> &mut T {
        &mut self.0[0][1]
    }

    /// Alias for `.get(2)`.
    ///
    /// # Panics
    /// When `N` < 3.
    pub fn z(&self) -> &T {
        &self.0[0][2]
    }

    pub fn z_mut(&mut self) -> &mut T {
        &mut self.0[0][2]
    }

    /// Alias for `.get(3)`.
    ///
    /// # Panics
    /// When `N` < 4.
    pub fn w(&self) -> &T {
        &self.0[0][3]
    }

    pub fn w_mut(&mut self) -> &mut T {
        &mut self.0[0][3]
    }

    /// Alias for `.x()`.
    pub fn r(&self) -> &T {
        self.x()
    }

    pub fn r_mut(&mut self) -> &mut T {
        self.x_mut()
    }

    /// Alias for `.y()`.
    pub fn g(&self) -> &T {
        self.y()
    }

    pub fn g_mut(&mut self) -> &mut T {
        self.y_mut()
    }

    /// Alias for `.z()`.
    pub fn b(&self) -> &T {
        self.z()
    }

    pub fn b_mut(&mut self) -> &mut T {
        self.z_mut()
    }

    /// Alias for `.w()`.
    pub fn a(&self) -> &T {
        self.w()
    }

    pub fn a_mut(&mut self) -> &mut T {
        self.w_mut()
    }
}

// Generates all the 2, 3, and 4-level swizzle functions.
#[cfg(feature = "swizzle")]
macro_rules! swizzle {
    // First level. Doesn't generate any functions itself because the one-letter functions
    // are manually provided in the Swizzle trait.
    ($a:ident, $x:ident, $y:ident, $z:ident, $w:ident) => {
        // Pass the alphabet so the second level can choose the next letters.
        swizzle!{ $a, $x, $x, $y, $z, $w }
        swizzle!{ $a, $y, $x, $y, $z, $w }
        swizzle!{ $a, $z, $x, $y, $z, $w }
        swizzle!{ $a, $w, $x, $y, $z, $w }
    };
    // Second level. Generates all 2-element swizzle functions, and recursively calls the
    // third level, specifying the third letter.
    ($a:ident, $b:ident, $x:ident, $y:ident, $z:ident, $w:ident) => {
        paste::item! {
            #[doc(hidden)]
            pub fn [< $a $b >](&self) -> ColumnVector<T, 2> {
                ColumnVector::<T, 2>::from([
                    self.$a().clone(),
                    self.$b().clone(),
                ])
            }
        }

        // Pass the alphabet so the third level can choose the next letters.
        swizzle!{ $a, $b, $x, $x, $y, $z, $w }
        swizzle!{ $a, $b, $y, $x, $y, $z, $w }
        swizzle!{ $a, $b, $z, $x, $y, $z, $w }
        swizzle!{ $a, $b, $w, $x, $y, $z, $w }
    };
    // Third level. Generates all 3-element swizzle functions, and recursively calls the
    // fourth level, specifying the fourth letter.
    ($a:ident, $b:ident, $c:ident, $x:ident, $y:ident, $z:ident, $w:ident) => {
        paste::item! {
            #[doc(hidden)]
            pub fn [< $a $b $c >](&self) -> ColumnVector<T, 3> {
                ColumnVector::<T, 3>::from([
                    self.$a().clone(),
                    self.$b().clone(),
                    self.$c().clone(),
                ])
            }
        }

        // Do not need to pass the alphabet because the fourth level does not need to choose
        // any more letters.
        swizzle!{ $a, $b, $c, $x }
        swizzle!{ $a, $b, $c, $y }
        swizzle!{ $a, $b, $c, $z }
        swizzle!{ $a, $b, $c, $w }
    };
    // Final level which halts the recursion. Generates all 4-element swizzle functions.
    // No $x, $y, $z, $w parameters because this function does not need to know the alphabet,
    // because it already has all the names assigned.
    ($a:ident, $b:ident, $c:ident, $d:ident) => {
        paste::item! {
            #[doc(hidden)]
            pub fn [< $a $b $c $d >](&self) -> ColumnVector<T, 4> {
                ColumnVector::<T, 4>::from([
                    self.$a().clone(),
                    self.$b().clone(),
                    self.$c().clone(),
                    self.$d().clone(),
                ])
            }
        }
    };
}

#[cfg(feature = "swizzle")]
impl<T, const N: usize> ColumnVector<T, N>
where
    T: Clone,
{
    swizzle! {x, x, y, z, w}
    swizzle! {y, x, y, z, w}
    swizzle! {z, x, y, z, w}
    swizzle! {w, x, y, z, w}
    swizzle! {r, r, g, b, a}
    swizzle! {g, r, g, b, a}
    swizzle! {b, r, g, b, a}
    swizzle! {a, r, g, b, a}
}

impl<T, const N: usize> VectorSpace for ColumnVector<T, N>
where
    T: Clone + Zero,
    T: Add<T, Output = T>,
    T: Sub<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Div<T, Output = T>,
{
    type Scalar = T;
}

impl<T, const N: usize> MetricSpace for ColumnVector<T, N>
where
    Self: InnerSpace,
{
    type Metric = <Self as VectorSpace>::Scalar;

    fn distance2(self, other: Self) -> Self::Metric {
        (other - self).magnitude2()
    }
}

impl<T, const N: usize> InnerSpace for ColumnVector<T, N>
where
    T: Clone + Zero,
    T: Add<T, Output = T>,
    T: Sub<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Div<T, Output = T>,
    Self: Clone,
{
    fn dot(self, rhs: Self) -> T {
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let mut sum = <T as Zero>::zero();
        let lhsp = &mut lhs as *mut MaybeUninit<Matrix<T, N, 1>> as *mut MaybeUninit<T>;
        let rhsp = &mut rhs as *mut MaybeUninit<Matrix<T, N, 1>> as *mut MaybeUninit<T>;
        for i in 0..N {
            sum = sum
                + unsafe {
                    lhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                        * rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                };
        }
        sum
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> From<ColumnVector<T, 2>> for mint::Vector2<T> {
    fn from(v: ColumnVector<T, 2>) -> mint::Vector2<T> {
        mint::Vector2 {
            x: *v.x(),
            y: *v.y(),
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector2<T>> for ColumnVector<T, 2> {
    fn from(mint_vec: mint::Vector2<T>) -> Self {
        Matrix([[mint_vec.x, mint_vec.y]])
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> From<ColumnVector<T, 3>> for mint::Vector3<T> {
    fn from(v: ColumnVector<T, 3>) -> mint::Vector3<T> {
        mint::Vector3 {
            x: *v.x(),
            y: *v.y(),
            z: *v.z(),
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector3<T>> for ColumnVector<T, 3> {
    fn from(mint_vec: mint::Vector3<T>) -> Self {
        Matrix([[mint_vec.x, mint_vec.y, mint_vec.z]])
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> From<ColumnVector<T, 4>> for mint::Vector4<T> {
    fn from(v: ColumnVector<T, 4>) -> mint::Vector4<T> {
        mint::Vector4 {
            x: *v.x(),
            y: *v.y(),
            z: *v.z(),
            w: *v.w(),
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector4<T>> for ColumnVector<T, 4> {
    fn from(mint_vec: mint::Vector4<T>) -> Self {
        Matrix([[mint_vec.x, mint_vec.y, mint_vec.z, mint_vec.w]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Vector1<T> = ColumnVector<T, 1>;
    type Vector2<T> = ColumnVector<T, 2>;
    type Vector3<T> = ColumnVector<T, 3>;
    type Vector4<T> = ColumnVector<T, 4>;

    #[test]
    fn test_zero() {
        let a = Vector3::<u32>::zero();
        assert_eq!(a, column_vector![0, 0, 0]);
    }

    #[test]
    fn test_index() {
        let a = Vector1::<u32>::from([0]);
        assert_eq!(*a.x(), 0_u32);
        let mut b = Vector2::<u32>::from([1, 2]);
        *b.y_mut() += 3;
        assert_eq!(*b.y(), 5);
    }

    #[test]
    fn test_eq() {
        let a = Vector1::<u32>::from([0]);
        let b = Vector1::<u32>::from([1]);
        let c = Vector1::<u32>::from([0]);
        let d = [[0u32]];
        assert_ne!(a, b);
        assert_eq!(a, c);
        assert_eq!(a, &d);
    }

    #[test]
    fn test_addition() {
        let a = Vector1::<u32>::from([0]);
        let b = Vector1::<u32>::from([1]);
        let c = Vector1::<u32>::from([2]);
        assert_eq!(a + b, b);
        assert_eq!(b + b, c);
        // We shouldn't need to have to test more dimensions, but we shall test
        // one more.
        let a = Vector2::<u32>::from([0, 1]);
        let b = Vector2::<u32>::from([1, 2]);
        let c = Vector2::<u32>::from([1, 3]);
        let d = Vector2::<u32>::from([2, 5]);
        assert_eq!(a + b, c);
        assert_eq!(b + c, d);
        let mut c = Vector2::<u32>::from([1, 3]);
        let d = Vector2::<u32>::from([2, 5]);
        c += d;
        let e = Vector2::<u32>::from([3, 8]);
        assert_eq!(c, e);
    }

    #[test]
    fn test_subtraction() {
        let mut a = Vector1::<u32>::from([3]);
        let b = Vector1::<u32>::from([1]);
        let c = Vector1::<u32>::from([2]);
        assert_eq!(a - c, b);
        a -= b;
        assert_eq!(a, c);
    }

    #[test]
    fn test_negation() {
        let a = Vector4::<i32>::from([1, 2, 3, 4]);
        let b = Vector4::<i32>::from([-1, -2, -3, -4]);
        assert_eq!(-a, b);
    }

    #[test]
    fn test_scale() {
        let a = Vector4::<f32>::from([2.0, 4.0, 2.0, 4.0]);
        let b = Vector4::<f32>::from([4.0, 8.0, 4.0, 8.0]);
        let c = Vector4::<f32>::from([1.0, 2.0, 1.0, 2.0]);
        assert_eq!(a * 2.0, b);
        assert_eq!(a / 2.0, c);
    }

    #[test]
    fn test_cross() {
        let a = column_vector!(1isize, 2isize, 3isize);
        let b = column_vector!(4isize, 5isize, 6isize);
        let r = column_vector!(-3isize, 6isize, -3isize);
        assert_eq!(a.cross(b), r);
    }

    #[test]
    fn test_distance() {
        let a = Vector1::<f32>::from([0.0]);
        let b = Vector1::<f32>::from([1.0]);
        assert_eq!(a.distance2(b), 1.0);
        let a = Vector1::<f32>::from([0.0]);
        let b = Vector1::<f32>::from([2.0]);
        assert_eq!(a.distance2(b), 4.0);
        assert_eq!(a.distance(b), 2.0);
        let a = Vector2::<f32>::from([0.0, 0.0]);
        let b = Vector2::<f32>::from([1.0, 1.0]);
        assert_eq!(a.distance2(b), 2.0);
    }

    #[test]
    fn test_normalize() {
        let a = column_vector!(5.0);
        assert_eq!(a.magnitude(), 5.0);
        let a_norm = a.normalize();
        assert_eq!(a_norm, column_vector!(1.0));
    }

    #[test]
    fn test_transpose() {
        let v = column_vector!(1i32, 2, 3, 4);
        let m = Matrix::<i32, 1, 4>::from([[1i32], [2], [3], [4]]);
        assert_eq!(v.transpose(), m);
    }

    #[test]
    fn test_from_fn() {
        let indices: ColumnVector<usize, 10> = column_vector!(0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        assert_eq!(ColumnVector::<usize, 10>::from_fn(|i| i), indices);
    }

    #[test]
    fn test_map() {
        let int = column_vector!(1i32, 0, 1, 1, 0, 1, 1, 0, 0, 0);
        let boolean =
            column_vector!(true, false, true, true, false, true, true, false, false, false);
        assert_eq!(int.map(|i| i != 0), boolean);
    }

    #[test]
    fn test_indexed_map() {
        let boolean =
            column_vector!(true, false, true, true, false, true, true, false, false, false);
        let indices = column_vector!(0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        assert_eq!(boolean.indexed_map(|i, _| i), indices);
    }

    #[test]
    fn test_from_iter() {
        let v = vec![1i32, 2, 3, 4];
        let vec = ColumnVector::<i32, 4>::from_iter(v);
        assert_eq!(vec, column_vector![1i32, 2, 3, 4])
    }

    #[test]
    fn test_linear_interpolate() {
        let v1 = column_vector!(0.0, 0.0, 0.0);
        let v2 = column_vector!(1.0, 2.0, 3.0);
        assert_eq!(v1.lerp(v2, 0.5), column_vector!(0.5, 1.0, 1.5));
    }

    #[test]
    fn test_reflect() {
        // Incident straight on to the surface.
        let v = column_vector!(1, 0);
        let n = column_vector!(-1, 0);
        let r = v.reflect(n);
        assert_eq!(r, column_vector!(-1, 0));

        // Incident at 45 degree angle to the surface.
        let v = column_vector!(1, 1);
        let n = column_vector!(-1, 0);
        let r = v.reflect(n);
        assert_eq!(r, column_vector!(-1, 1));
    }
}
