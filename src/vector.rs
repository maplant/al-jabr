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
pub type Vector<T, const N: usize> = Matrix<T, N, 1>;
//pub struct Vector<T, const N: usize>(pub(crate) [T; N]);

impl<T, const N: usize> Vector<T, N> {
    /// Convert the Vector into its inner array.
    pub fn into_inner(self) -> [T; N] {
        let Matrix([inner]) = self;
        inner
    }

    /// Constructs a new vector whose elements are equal to the value of the
    /// given function evaluated at the element's index.
    pub fn from_fn<Out, F>(mut f: F) -> Vector<Out, { N }>
    where
        F: FnMut(usize) -> Out,
    {
        let mut to = MaybeUninit::<Vector<Out, { N }>>::uninit();
        let top: *mut Out = unsafe { mem::transmute(&mut to) };
        for i in 0..N {
            unsafe { top.add(i).write(f(i)) }
        }
        unsafe { to.assume_init() }
    }

    pub fn indexed_map<Out, F>(self, mut f: F) -> Vector<Out, { N }>
    where
        F: FnMut(usize, T) -> Out,
    {
        let mut from = MaybeUninit::new(self);
        let mut to = MaybeUninit::<Vector<Out, { N }>>::uninit();
        let fromp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut from) };
        let top: *mut Out = unsafe { mem::transmute(&mut to) };
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

impl<T, const N: usize> Vector<T, { N }>
where
    T: Clone,
{
    /// Returns the first `M` elements of `self` in an appropriately sized
    /// `Vector`.
    ///
    /// Calling `first` with `M > N` is a compile error.
    pub fn first<const M: usize>(&self) -> Vector<T, { M }> {
        if M > N {
            panic!("attempt to return {} elements from a {}-vector", M, N);
        }
        let mut head = MaybeUninit::<Vector<T, { M }>>::uninit();
        let headp: *mut T = unsafe { mem::transmute(&mut head) };
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
    pub fn last<const M: usize>(&self) -> Vector<T, { M }> {
        if M > N {
            panic!("attempt to return {} elements from a {}-vector", M, N);
        }
        let mut tail = MaybeUninit::<Vector<T, { M }>>::uninit();
        let tailp: *mut T = unsafe { mem::transmute(&mut tail) };
        for i in 0..M {
            unsafe {
                tailp.add(i + N - M).write(self.0[0][i].clone());
            }
        }
        unsafe { tail.assume_init() }
    }
}

impl<T> Vector3<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Clone,
{
    /// Return the cross product of the two vectors.
    pub fn cross(self, rhs: Vector3<T>) -> Self {
        let Matrix([[x0, y0, z0]]) = self;
        let Matrix([[x1, y1, z1]]) = rhs;
        Vector3::from([[
            (y0.clone() * z1.clone()) - (z0.clone() * y1.clone()),
            (z0 * x1.clone()) - (x0.clone() * z1),
            (x0 * y1) - (y0 * x1),
        ]])
    }
}

impl<T, const N: usize> Vector<T, { N }>
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

impl<T, const N: usize> From<[T; N]> for Vector<T, { N }> {
    fn from(array: [T; N]) -> Self {
        Matrix([array])
    }
}

impl<T, const N: usize> Into<[T; N]> for Vector<T, N> {
    fn into(self: Vector<T, N>) -> [T; N] {
        let Matrix([vec]) = self;
        vec
    }
}

/// 2-element vector.
pub type Vector2<T> = Vector<T, 2>;

/// 3-element vector.
pub type Vector3<T> = Vector<T, 3>;

/// 4-element vector.
pub type Vector4<T> = Vector<T, 4>;

/// Construct a new [Vector] of any size.
///
/// ```
/// # use al_jabr::*;
/// let v: Vector<u32, 0> = vector![];
/// let v = vector![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let v = vector![true, false, false, true];
/// ```
#[macro_export]
macro_rules! vector {
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
impl<T, const N: usize> Vector<T, { N }> {
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
            pub fn [< $a $b >](&self) -> Vector<T, 2> {
                Vector::<T, 2>::from([
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
            pub fn [< $a $b $c >](&self) -> Vector<T, 3> {
                Vector::<T, 3>::from([
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
            pub fn [< $a $b $c $d >](&self) -> Vector<T, 4> {
                Vector::<T, 4>::from([
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
impl<T, const N: usize> Vector<T, { N }>
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

impl<T, const N: usize> VectorSpace for Vector<T, { N }>
where
    T: Clone + Zero,
    T: Add<T, Output = T>,
    T: Sub<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Div<T, Output = T>,
{
    type Scalar = T;
}

impl<T, const N: usize> MetricSpace for Vector<T, { N }>
where
    Self: InnerSpace,
{
    type Metric = <Self as VectorSpace>::Scalar;

    fn distance2(self, other: Self) -> Self::Metric {
        (other - self).magnitude2()
    }
}

impl<T, const N: usize> InnerSpace for Vector<T, { N }>
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
        let lhsp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut lhs) };
        let rhsp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            sum = sum + unsafe {
                lhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                    * rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
            };
        }
        sum
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> Into<mint::Vector2<T>> for Vector<T, 2> {
    fn into(self) -> mint::Vector2<T> {
        mint::Vector2 {
            x: self.0[0][0],
            y: self.0[0][1],
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector2<T>> for Vector<T, 2> {
    fn from(mint_vec: mint::Vector2<T>) -> Self {
        Matrix([[mint_vec.x, mint_vec.y]])
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> Into<mint::Vector3<T>> for Vector<T, 3> {
    fn into(self) -> mint::Vector3<T> {
        mint::Vector3 {
            x: self.0[0][0],
            y: self.0[0][1],
            z: self.0[0][2],
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector3<T>> for Vector<T, 3> {
    fn from(mint_vec: mint::Vector3<T>) -> Self {
        Matrix([[mint_vec.x, mint_vec.y, mint_vec.z]])
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> Into<mint::Vector4<T>> for Vector<T, 4> {
    fn into(self) -> mint::Vector4<T> {
        mint::Vector4 {
            x: self.0[0][0],
            y: self.0[0][1],
            z: self.0[0][2],
            w: self.0[0][3],
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector4<T>> for Vector<T, 4> {
    fn from(mint_vec: mint::Vector4<T>) -> Self {
        Matrix([[mint_vec.x, mint_vec.y, mint_vec.z, mint_vec.w]])
    }
}
