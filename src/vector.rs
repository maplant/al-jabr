//! `N`-element vector.
use super::*;

/// `N`-element vector.
///
/// Vectors can be constructed from arrays of any type and size. There are
/// convenience constructor functions provided for the most common sizes.
///
/// ```
/// # use aljabar::*;
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
# use aljabar::*;
let z = vector!(1i32, 2).z(); // Will panic.
```
### Mixing
zle methods are not implemented for mixed xyzw/rgba methods.
```
# use aljabar::*;
let v = vector!(1i32, 2, 3, 4);
let xy = v.xy(); // OK, only uses xyzw names.
let ba = v.ba(); // OK, only uses rgba names.
assert_eq!(xy, vector!(1i32, 2));
assert_eq!(ba, vector!(3i32, 4));
```
```compile_fail
# use aljabar::*;
let v = vector!(1i32, 2, 3, 4);
let bad = v.xyrg(); // Compile error, mixes xyzw and rgba names.
```
## Examples
To get the first two elements of a 4-vector.
```
# use aljabar::*;
let v = vector!(1i32, 2, 3, 4).xy();
```
To get the first and last element of a 4-vector.
```
# use aljabar::*;
let v = vector!(1i32, 2, 3, 4).xw();
```
To reverse the order of a 3-vector.
```
# use aljabar::*;
let v = vector!(1i32, 2, 3).zyx();
```
To select the first and third elements into the second and fourth elements,
respectively.
```
# use aljabar::*;
let v = vector!(1i32, 2, 3, 4).xxzz();
```
"##
)]
#[repr(transparent)]
pub struct Vector<T, const N: usize>(pub(crate) [T; N]);

impl<T, const N: usize> Vector<T, { N }> {
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
    /// Applies the given function to each element of the vector, constructing a
    /// new vector with the returned outputs.
    pub fn map<Out, F>(self, mut f: F) -> Vector<Out, { N }>
    where
        F: FnMut(T) -> Out,
    {
        self.indexed_map(|_, x: T| -> Out { f(x) })
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

    /// Converts the Vector into a Matrix with `N` columns each of size `1`.
    ///
    /// ```ignore
    /// # use aljabar::*;
    /// let v = vector!(1i32, 2, 3, 4);
    /// let m = Matrix::<i32, 1, 4>::from([
    ///     vector!(1i32),
    ///     vector!(2),
    ///     vector!(3),
    ///     vector!(4),
    /// ]);
    /// assert_eq!(v.transpose(), m);
    /// ```
    pub fn transpose(self) -> Matrix<T, 1, { N }> {
        let mut from = MaybeUninit::new(self);
        let mut st = MaybeUninit::<Matrix<T, 1, { N }>>::uninit();
        let fromp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut from) };
        let stp: *mut Vector<T, 1> = unsafe { mem::transmute(&mut st) };
        for i in 0..N {
            unsafe {
                stp.add(i).write(Vector::<T, 1>::from([fromp
                    .add(i)
                    .replace(MaybeUninit::uninit())
                    .assume_init()]));
            }
        }
        unsafe { st.assume_init() }
    }

    /// Removes the last component and returns the vector with one fewer
    /// dimension.
    ///
    /// ```
    /// # use aljabar::*;
    /// let (xyz, w) = vector!(0u32, 1, 2, 3).truncate();
    /// assert_eq!(xyz, vector!(0u32, 1, 2));
    /// assert_eq!(w, 3);
    /// ```
    pub fn truncate(self) -> (TruncatedVector<T, { N }>, T) {
        let mut from = MaybeUninit::new(self);
        let mut head = MaybeUninit::<TruncatedVector<T, { N }>>::uninit();
        let fromp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut from) };
        let headp: *mut T = unsafe { mem::transmute(&mut head) };
        for i in 0..(N - 1) {
            unsafe {
                headp
                    .add(i)
                    .write(fromp.add(i).replace(MaybeUninit::uninit()).assume_init());
            }
        }
        (unsafe { head.assume_init() }, unsafe {
            fromp
                .add(N - 1)
                .replace(MaybeUninit::uninit())
                .assume_init()
        })
    }

    /// Extends the vector with an additional value.
    ///
    /// Useful for performing affine transformations.
    /// ```
    /// # use aljabar::*;
    /// let xyzw = vector!(0u32, 1, 2).extend(3);
    /// assert_eq!(xyzw, vector!(0u32, 1, 2, 3));
    /// ```
    pub fn extend(self, new: T) -> ExtendedVector<T, { N }> {
        let mut from = MaybeUninit::new(self);
        let mut head = MaybeUninit::<ExtendedVector<T, { N }>>::uninit();
        let fromp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut from) };
        let headp: *mut T = unsafe { mem::transmute(&mut head) };
        for i in 0..N {
            unsafe {
                headp
                    .add(i)
                    .write(fromp.add(i).replace(MaybeUninit::uninit()).assume_init());
            }
        }
        unsafe {
            headp.add(N).write(new);
            head.assume_init()
        }
    }
}

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
                headp.add(i).write(self[i].clone());
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
                tailp.add(i + N - M).write(self[i].clone());
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
        let [x0, y0, z0]: [T; 3] = self.into();
        let [x1, y1, z1]: [T; 3] = rhs.into();
        Vector3::from([
            (y0.clone() * z1.clone()) - (z0.clone() * y1.clone()),
            (z0 * x1.clone()) - (x0.clone() * z1),
            (x0 * y1) - (y0 * x1),
        ])
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
        let mut v_max = self.0[0].clone();
        for i in 1..N {
            if self.0[i] > v_max {
                i_max = i;
                v_max = self.0[i].clone();
            }
        }
        (i_max, v_max)
    }

    /// Return the largest value in the vector.
    pub fn max(&self) -> T {
        let mut v_max = self.0[0].clone();
        for i in 1..N {
            if self.0[i] > v_max {
                v_max = self.0[i].clone();
            }
        }
        v_max
    }

    /// Return the smallest value found in the vector, along with the
    /// associated index.
    pub fn argmin(&self) -> (usize, T) {
        let mut i_min = 0;
        let mut v_min = self.0[0].clone();
        for i in 1..N {
            if self.0[i] < v_min {
                i_min = i;
                v_min = self.0[i].clone();
            }
        }
        (i_min, v_min)
    }

    /// Return the smallest value in the vector.
    pub fn min(&self) -> T {
        let mut v_min = self.0[0].clone();
        for i in 1..N {
            if self.0[i] < v_min {
                v_min = self.0[i].clone();
            }
        }
        v_min
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, { N }> {
    fn from(array: [T; N]) -> Self {
        Vector::<T, { N }>(array)
    }
}

impl<T, const N: usize> From<Matrix<T, { N }, 1>> for Vector<T, { N }> {
    fn from(mat: Matrix<T, { N }, 1>) -> Self {
        let Matrix([v]) = mat;
        v
    }
}

/// 2-element vector.
pub type Vector2<T> = Vector<T, 2>;

/// 3-element vector.
pub type Vector3<T> = Vector<T, 3>;

/// 4-element vector.
pub type Vector4<T> = Vector<T, 4>;

/// Constructs a new vector from an array. Necessary to help the compiler.
/// Prefer calling the macro `vector!`, which calls `new_vector` internally.
#[inline]
#[doc(hidden)]
pub fn new_vector<T, const N: usize>(elements: [T; N]) -> Vector<T, { N }> {
    Vector(elements)
}

/// Construct a new [Vector] of any size.
///
/// ```
/// # use aljabar::*;
/// let v: Vector<u32, 0> = vector![];
/// let v = vector![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let v = vector![true, false, false, true];
/// ```
#[macro_export]
macro_rules! vector {
    ( $($elem:expr),* $(,)? ) => {
        $crate::new_vector([
            $($elem),*
        ])
    }
}

impl<T, const N: usize> Clone for Vector<T, { N }>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Vector::<T, { N }>(self.0.clone())
    }
}

impl<T, const N: usize> Copy for Vector<T, { N }> where T: Copy {}

impl<T, const N: usize> Into<[T; { N }]> for Vector<T, { N }> {
    fn into(self) -> [T; { N }] {
        self.0
    }
}

impl<T, const N: usize> fmt::Debug for Vector<T, { N }>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match N {
            0 => unimplemented!(),
            1 => write!(f, "Vector {{ x: {:?} }}", self.0[0]),
            2 => write!(f, "Vector {{ x: {:?}, y: {:?} }}", self.0[0], self.0[1]),
            3 => write!(
                f,
                "Vector {{ x: {:?}, y: {:?}, z: {:?} }}",
                self.0[0], self.0[1], self.0[2]
            ),
            4 => write!(
                f,
                "Vector {{ x: {:?}, y: {:?}, z: {:?}, w: {:?} }}",
                self.0[0], self.0[1], self.0[2], self.0[3]
            ),
            _ => write!(
                f,
                "Vector {{ x: {:?}, y: {:?}, z: {:?}, w: {:?}, [..]: {:?} }}",
                self.0[0],
                self.0[1],
                self.0[2],
                self.0[3],
                &self.0[4..]
            ),
        }
    }
}

impl<T, const N: usize> Deref for Vector<T, { N }> {
    type Target = [T; { N }];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for Vector<T, { N }> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const N: usize> Hash for Vector<T, { N }>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in 0..N {
            self.0[i].hash(state);
        }
    }
}

impl<T, const N: usize> FromIterator<T> for Vector<T, { N }> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let mut new = MaybeUninit::<Vector<T, { N }>>::uninit();
        let newp: *mut T = unsafe { mem::transmute(&mut new) };

        for i in 0..N {
            if let Some(next) = iter.next() {
                unsafe { newp.add(i).write(next) };
            } else {
                panic!("too few items in iterator to create Vector<_, {}>", N);
            }
        }

        if iter.next().is_some() {
            panic!("too many items in iterator to create Vector<_, {}>", N);
        }

        unsafe { new.assume_init() }
    }
}

impl<T, const N: usize> IntoIterator for Vector<T, { N }> {
    type Item = T;
    type IntoIter = ArrayIter<T, { N }>;

    fn into_iter(self) -> Self::IntoIter {
        let Vector(array) = self;
        ArrayIter {
            array: MaybeUninit::new(array),
            pos:   0,
        }
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
impl<T, const N: usize> Vector<T, { N }>
where
    T: Clone,
{
    /// Alias for `.get(0).clone()`.
    ///
    /// # Panics
    /// When `N` = 0.
    pub fn x(&self) -> T {
        self.0[0].clone()
    }

    /// Alias for `.get(1).clone()`.
    ///
    /// # Panics
    /// When `N` < 2.
    pub fn y(&self) -> T {
        self.0[1].clone()
    }

    /// Alias for `.get(2).clone()`.
    ///
    /// # Panics
    /// When `N` < 3.
    pub fn z(&self) -> T {
        self.0[2].clone()
    }

    /// Alias for `.get(3).clone()`.
    ///
    /// # Panics
    /// When `N` < 4.
    pub fn w(&self) -> T {
        self.0[3].clone()
    }

    /// Alias for `.x()`.
    pub fn r(&self) -> T {
        self.x()
    }

    /// Alias for `.y()`.
    pub fn g(&self) -> T {
        self.y()
    }

    /// Alias for `.z()`.
    pub fn b(&self) -> T {
        self.z()
    }

    /// Alias for `.w()`.
    pub fn a(&self) -> T {
        self.w()
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
                    self.$a(),
                    self.$b(),
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
                    self.$a(),
                    self.$b(),
                    self.$c(),
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
                    self.$a(),
                    self.$b(),
                    self.$c(),
                    self.$d(),
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

impl<T, const N: usize> Zero for Vector<T, { N }>
where
    T: Zero,
{
    fn zero() -> Self {
        let mut origin = MaybeUninit::<Vector<T, { N }>>::uninit();
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

impl<A, B, RHS, const N: usize> PartialEq<RHS> for Vector<A, { N }>
where
    RHS: Deref<Target = [B; { N }]>,
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

impl<T, const N: usize> Eq for Vector<T, { N }> where T: Eq {}

impl<A, B, const N: usize> Add<Vector<B, { N }>> for Vector<A, { N }>
where
    A: Add<B>,
{
    type Output = Vector<<A as Add<B>>::Output, { N }>;

    fn add(self, rhs: Vector<B, { N }>) -> Self::Output {
        let mut sum = MaybeUninit::<[<A as Add<B>>::Output; { N }]>::uninit();
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let sump: *mut <A as Add<B>>::Output = unsafe { mem::transmute(&mut sum) };
        let lhsp: *mut MaybeUninit<A> = unsafe { mem::transmute(&mut lhs) };
        let rhsp: *mut MaybeUninit<B> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            unsafe {
                sump.add(i).write(
                    lhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                        + rhsp.add(i).replace(MaybeUninit::uninit()).assume_init(),
                );
            }
        }
        Vector::<<A as Add<B>>::Output, { N }>(unsafe { sum.assume_init() })
    }
}

impl<A, B, const N: usize> AddAssign<Vector<B, { N }>> for Vector<A, { N }>
where
    A: AddAssign<B>,
{
    fn add_assign(&mut self, rhs: Vector<B, { N }>) {
        let mut rhs = MaybeUninit::new(rhs);
        let rhsp: *mut MaybeUninit<B> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            self.0[i] += unsafe { rhsp.add(i).replace(MaybeUninit::uninit()).assume_init() };
        }
    }
}

impl<A, B, const N: usize> Sub<Vector<B, { N }>> for Vector<A, { N }>
where
    A: Sub<B>,
{
    type Output = Vector<<A as Sub<B>>::Output, { N }>;

    fn sub(self, rhs: Vector<B, { N }>) -> Self::Output {
        let mut dif = MaybeUninit::<[<A as Sub<B>>::Output; { N }]>::uninit();
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let difp: *mut <A as Sub<B>>::Output = unsafe { mem::transmute(&mut dif) };
        let lhsp: *mut MaybeUninit<A> = unsafe { mem::transmute(&mut lhs) };
        let rhsp: *mut MaybeUninit<B> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            unsafe {
                difp.add(i).write(
                    lhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                        - rhsp.add(i).replace(MaybeUninit::uninit()).assume_init(),
                );
            }
        }
        Vector::<<A as Sub<B>>::Output, { N }>(unsafe { dif.assume_init() })
    }
}

impl<A, B, const N: usize> SubAssign<Vector<B, { N }>> for Vector<A, { N }>
where
    A: SubAssign<B>,
{
    fn sub_assign(&mut self, rhs: Vector<B, { N }>) {
        let mut rhs = MaybeUninit::new(rhs);
        let rhsp: *mut MaybeUninit<B> = unsafe { mem::transmute(&mut rhs) };
        for i in 0..N {
            self.0[i] -= unsafe { rhsp.add(i).replace(MaybeUninit::uninit()).assume_init() };
        }
    }
}

impl<T, const N: usize> Neg for Vector<T, { N }>
where
    T: Neg,
{
    type Output = Vector<<T as Neg>::Output, { N }>;

    fn neg(self) -> Self::Output {
        let mut from = MaybeUninit::new(self);
        let mut neg = MaybeUninit::<[<T as Neg>::Output; { N }]>::uninit();
        let fromp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut from) };
        let negp: *mut <T as Neg>::Output = unsafe { mem::transmute(&mut neg) };
        for i in 0..N {
            unsafe {
                negp.add(i).write(
                    fromp
                        .add(i)
                        .replace(MaybeUninit::uninit())
                        .assume_init()
                        .neg(),
                );
            }
        }
        Vector::<<T as Neg>::Output, { N }>(unsafe { neg.assume_init() })
    }
}

/// Scalar multiply
impl<A, B, const N: usize> Mul<B> for Vector<A, { N }>
where
    A: Mul<B>,
    B: Clone,
{
    type Output = Vector<<A as Mul<B>>::Output, { N }>;

    fn mul(self, scalar: B) -> Self::Output {
        let mut from = MaybeUninit::new(self);
        let mut scaled = MaybeUninit::<[<A as Mul<B>>::Output; { N }]>::uninit();
        let fromp: *mut MaybeUninit<A> = unsafe { mem::transmute(&mut from) };
        let scaledp: *mut <A as Mul<B>>::Output = unsafe { mem::transmute(&mut scaled) };
        for i in 0..N {
            unsafe {
                scaledp.add(i).write(
                    fromp.add(i).replace(MaybeUninit::uninit()).assume_init() * scalar.clone(),
                );
            }
        }
        Vector::<<A as Mul<B>>::Output, { N }>(unsafe { scaled.assume_init() })
    }
}

impl<const N: usize> Mul<Vector<f32, { N }>> for f32 {
    type Output = Vector<f32, { N }>;

    fn mul(self, vec: Vector<f32, { N }>) -> Self::Output {
        vec * self
    }
}

impl<const N: usize> Mul<Vector<f64, { N }>> for f64 {
    type Output = Vector<f64, { N }>;

    fn mul(self, vec: Vector<f64, { N }>) -> Self::Output {
        vec * self
    }
}

/// Scalar multiply assign
impl<A, B, const N: usize> MulAssign<B> for Vector<A, { N }>
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
impl<A, B, const N: usize> Div<B> for Vector<A, { N }>
where
    A: Div<B>,
    B: Clone,
{
    type Output = Vector<<A as Div<B>>::Output, { N }>;

    fn div(self, scalar: B) -> Self::Output {
        let mut from = MaybeUninit::new(self);
        let mut scaled = MaybeUninit::<[<A as Div<B>>::Output; { N }]>::uninit();
        let fromp: *mut MaybeUninit<A> = unsafe { mem::transmute(&mut from) };
        let scaledp: *mut <A as Div<B>>::Output = unsafe { mem::transmute(&mut scaled) };
        for i in 0..N {
            unsafe {
                scaledp.add(i).write(
                    fromp.add(i).replace(MaybeUninit::uninit()).assume_init() / scalar.clone(),
                );
            }
        }
        Vector::<<A as Div<B>>::Output, { N }>(unsafe { scaled.assume_init() })
    }
}

/// Scalar divide assign
impl<A, B, const N: usize> DivAssign<B> for Vector<A, { N }>
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
                lhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
                    * rhsp.add(i).replace(MaybeUninit::uninit()).assume_init()
            };
        }
        sum
    }
}

#[cfg(feature = "rand")]
impl<T, const N: usize> Distribution<Vector<T, { N }>> for Standard
where
    Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector<T, { N }> {
        let mut rand = MaybeUninit::<Vector<T, { N }>>::uninit();
        let randp: *mut T = unsafe { mem::transmute(&mut rand) };

        for i in 0..N {
            unsafe { randp.add(i).write(self.sample(rng)) }
        }

        unsafe { rand.assume_init() }
    }
}

#[cfg(feature = "serde")]
impl<T, const N: usize> Serialize for Vector<T, { N }>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_tuple(N)?;
        for i in 0..N {
            seq.serialize_element(&self.0[i])?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, const N: usize> Deserialize<'de> for Vector<T, { N }>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_tuple(N, ArrayVisitor::<[T; { N }]>::new())
            .map(Vector)
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> Into<mint::Vector2<T>> for Vector<T, 2> {
    fn into(self) -> mint::Vector2<T> {
        mint::Vector2 {
            x: self.0[0],
            y: self.0[1],
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector2<T>> for Vector<T, 2> {
    fn from(mint_vec: mint::Vector2<T>) -> Self {
        Vector([mint_vec.x, mint_vec.y])
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> Into<mint::Vector3<T>> for Vector<T, 3> {
    fn into(self) -> mint::Vector3<T> {
        mint::Vector3 {
            x: self.0[0],
            y: self.0[1],
            z: self.0[2],
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector3<T>> for Vector<T, 3> {
    fn from(mint_vec: mint::Vector3<T>) -> Self {
        Vector([mint_vec.x, mint_vec.y, mint_vec.z])
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> Into<mint::Vector4<T>> for Vector<T, 4> {
    fn into(self) -> mint::Vector4<T> {
        mint::Vector4 {
            x: self.0[0],
            y: self.0[1],
            z: self.0[2],
            w: self.0[3],
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector4<T>> for Vector<T, 4> {
    fn from(mint_vec: mint::Vector4<T>) -> Self {
        Vector([mint_vec.x, mint_vec.y, mint_vec.z, mint_vec.w])
    }
}
