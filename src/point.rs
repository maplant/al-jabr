//! A point in space.
//!
//! Points are far less flexible and useful than vectors and are used
//! to express the purpose or meaning of a variable through its type.
//!
//! Points can be moved through space by adding or subtracting
//! [vectors](vector) from them.
//!
//! The only mathematical operator supported between two points is
//! subtraction, which results in the vector between the two points.
//!
//! Points can be freely converted to and from vectors via `From`.

use super::*;

/// A point in 2-dimensional space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point2<T> {
    /// Construct a new point 
    pub fn new(x: T, y: T) -> Self {
        Self {
            x,
            y
        }
    }

    /// Convert a point from a [Vector2]
    pub fn from_vec(vec: Vector2<T>) -> Self {
        Self {
            x: vec.x,
            y: vec.y,
        }
    }

    /// Convert the point into a [Vector2]
    pub fn to_vec(self) -> Vector2<T> {
        Vector2 {
            x: self.x,
            y: self.y,
        }
    }

    /// Construct a Point from a [ColumnVector]
    pub fn from_col(Matrix([[x, y]]): ColumnVector<T, 2>) -> Self {
        Self {
            x,
            y
        }
    }

    /// Convert the point into a [ColumnVector]
    pub fn to_col(self) -> ColumnVector<T, 2> {
        Matrix([[self.x, self.y]])
    }

    /// Construct a new point by mapping the components of the old point.
    pub fn map<B>(self, mut mapper: impl FnMut(T) -> B) -> Point2<B> {
        Point2 {
            x: mapper(self.x),
            y: mapper(self.y),
        }
    }

    /// Construct a point where each element is the pair of the components
    /// of the two points.
    pub fn zip<B>(self, p2: Point2<B>) -> Point2<(T, B)> {
        Point2 {
            x: (self.x, p2.x),
            y: (self.y, p2.y),
        }
    }

    /// Iterate over the point, in x, y, z order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.x, &self.y].into_iter()
    }

    /// Mutably iterate over the point, in x, y, z order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        [&mut self.x, &mut self.y].into_iter()
    }

    /// Convert the point into an iterator, in x, y, z order.
    pub fn into_iter(self) -> impl IntoIterator<Item = T> {
        [self.x, self.y].into_iter()
    }
}

impl<T: Zero> Point2<T> {
    pub fn origin() -> Self {
        Self {
            x: T::zero(),
            y: T::zero(),
        }
    }
}

impl<A, B> Add<Vector2<B>> for Point2<A>
where
    A: Add<B>,
{
    type Output = Point2<A::Output>;

    fn add(self, rhs: Vector2<B>) -> Self::Output {
        Point2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<A, B> Sub<Vector2<B>> for Point2<A>
where
    A: Sub<B>,
{
    type Output = Point2<A::Output>;

    fn sub(self, rhs: Vector2<B>) -> Self::Output {
        Point2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<A, B> Sub<Point2<B>> for Point2<A>
where
    A: Sub<B>,
{
    type Output = Vector2<A::Output>;

    fn sub(self, rhs: Point2<B>) -> Self::Output {
        Vector2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

#[cfg(any(feature = "approx", test))]
impl<T: AbsDiffEq> AbsDiffEq for Point2<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        T::abs_diff_eq(&self.x, &other.x, epsilon) &&
            T::abs_diff_eq(&self.y, &other.y, epsilon)
    }
}

#[cfg(feature = "approx")]
impl<T> RelativeEq for Point2<T>
where
    T: RelativeEq,
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        T::relative_eq(&self.x, &other.x, epsilon, max_relative) &&
            T::relative_eq(&self.y, &other.y, epsilon, max_relative)
    }
}

#[cfg(feature = "approx")]
impl<T> UlpsEq for Point2<T>
where
    T: UlpsEq,
    T::Epsilon: Copy,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        T::ulps_eq(&self.x, &other.x, epsilon, max_ulps) &&
            T::ulps_eq(&self.y, &other.y, epsilon, max_ulps)
    }
}

#[cfg(feature = "rand")]
impl<T> Distribution<Point2<T>> for Standard
where
    Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point2<T> {
        Self {
            x: self.sample(rng),
            y: self.sample(rng),
        }
    }
}

/// A point in 3-dimensional space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point3<T> {
    pub fn from_vec(vec: Vector3<T>) -> Self {
        Self {
            x: vec.x,
            y: vec.y,
            z: vec.z,
        }
    }

    pub fn to_vec(self) -> Vector3<T> {
        Vector3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
    
    pub fn from_col(Matrix([[x, y, z]]): ColumnVector<T, 3>) -> Self {
        Self {
            x,
            y,
            z
        }
    }

    pub fn to_col(self) -> ColumnVector<T, 3> {
        Matrix([[self.x, self.y, self.z]])
    }
}

impl<T: Zero> Point3<T> {
    pub fn origin() -> Self {
        Self {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }
}

impl<A, B> Add<Vector3<B>> for Point3<A>
where
    A: Add<B>,
{
    type Output = Point3<A::Output>;

    fn add(self, rhs: Vector3<B>) -> Self::Output {
        Point3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<A, B> Sub<Vector3<B>> for Point3<A>
where
    A: Sub<B>,
{
    type Output = Point3<A::Output>;

    fn sub(self, rhs: Vector3<B>) -> Self::Output {
        Point3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<A, B> Sub<Point3<B>> for Point3<A>
where
    A: Sub<B>,
{
    type Output = Vector3<A::Output>;

    fn sub(self, rhs: Point3<B>) -> Self::Output {
        Vector3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

#[cfg(any(feature = "approx", test))]
impl<T: AbsDiffEq> AbsDiffEq for Point3<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        T::abs_diff_eq(&self.x, &other.x, epsilon) &&
            T::abs_diff_eq(&self.y, &other.y, epsilon) &&
            T::abs_diff_eq(&self.z, &other.z, epsilon)
    }
}


#[cfg(feature = "approx")]
impl<T> RelativeEq for Point3<T>
where
    T: RelativeEq,
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        T::relative_eq(&self.x, &other.x, epsilon, max_relative) &&
            T::relative_eq(&self.y, &other.y, epsilon, max_relative) &&
            T::relative_eq(&self.z, &other.z, epsilon, max_relative)
    }
}

#[cfg(feature = "approx")]
impl<T> UlpsEq for Point3<T>
where
    T: UlpsEq,
    T::Epsilon: Copy,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        T::ulps_eq(&self.x, &other.x, epsilon, max_ulps) &&
            T::ulps_eq(&self.y, &other.y, epsilon, max_ulps)  &&
            T::ulps_eq(&self.z, &other.z, epsilon, max_ulps) 
    }
}

#[cfg(feature = "rand")]
impl<T> Distribution<Point3<T>> for Standard
where
    Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point3<T> {
        Self {
            x: self.sample(rng),
            y: self.sample(rng),
            z: self.sample(rng),
        }
    }
}

/*
/// A point in space.
#[repr(transparent)]
pub struct Point<T, const N: usize>(pub(crate) [T; N]);

impl<T, const N: usize> Point<T, N> {
    /// Convenience method for converting from vector.
    pub fn from_vec(Matrix([v]): ColumnVector<T, N>) -> Self {
        Self(v)
    }

    /// Convenience method for converting from vector.
    pub fn to_vec(self) -> ColumnVector<T, N> {
        Matrix([self.0])
    }
}

impl<T, const N: usize> Point<T, N>
where
    T: Zero,
{
    /// Construct a point at the origin.
    pub fn origin() -> Self {
        Self::from_vec(ColumnVector::zero())
    }
}

impl<T, const N: usize> Point<T, N> {
    /// Alias for `.get(0)`.
    ///
    /// # Panics
    /// When `N` = 0.
    pub fn x(&self) -> &T {
        &self.0[0]
    }

    pub fn x_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }

    /// Alias for `.get(1)`.
    ///
    /// # Panics
    /// When `N` < 2.
    pub fn y(&self) -> &T {
        &self.0[1]
    }

    pub fn y_mut(&mut self) -> &mut T {
        &mut self.0[1]
    }

    /// Alias for `.get(2)`.
    ///
    /// # Panics
    /// When `N` < 3.
    pub fn z(&self) -> &T {
        &self.0[2]
    }

    pub fn z_mut(&mut self) -> &mut T {
        &mut self.0[2]
    }

    /// Alias for `.get(3)`.
    ///
    /// # Panics
    /// When `N` < 4.
    pub fn w(&self) -> &T {
        &self.0[3]
    }

    pub fn w_mut(&mut self) -> &mut T {
        &mut self.0[3]
    }
}

impl<T, const N: usize> From<[T; N]> for Point<T, N> {
    fn from(array: [T; N]) -> Self {
        Point::<T, N>(array)
    }
}

impl<T, const N: usize> From<ColumnVector<T, N>> for Point<T, N> {
    fn from(Matrix([array]): ColumnVector<T, N>) -> Self {
        Point::<T, N>(array)
    }
}

/// A point in 1-dimensional space.
pub type Point1<T> = Point<T, 1>;

/// A point in 2-dimensional space.
pub type Point2<T> = Point<T, 2>;

impl<T> Point2<T> {
    /// Extend a Point1 into a Point2.
    pub fn from_point1(p: Point1<T>, y: T) -> Self {
        let Point([x]) = p;
        Point([x, y])
    }
}

/// A point in 3-dimensional space.
pub type Point3<T> = Point<T, 3>;

impl<T> Point3<T> {
    /// Extend a Point1 into a Point3.
    pub fn from_point1(p: Point1<T>, y: T, z: T) -> Self {
        let Point([x]) = p;
        Point([x, y, z])
    }

    /// Extend a Point2 into a Point3.
    pub fn from_point2(p: Point2<T>, z: T) -> Self {
        let Point([x, y]) = p;
        Point([x, y, z])
    }
}

/// A point in 4-dimensional space.
pub type Point4<T> = Point<T, 4>;

impl<T> Point4<T> {
    /// Extend a Point1 into a Point4.
    pub fn from_point1(p: Point1<T>, y: T, z: T, w: T) -> Self {
        let Point([x]) = p;
        Point([x, y, z, w])
    }

    /// Extend a Point2 into a Point4.
    pub fn from_point2(p: Point2<T>, z: T, w: T) -> Self {
        let Point([x, y]) = p;
        Point([x, y, z, w])
    }

    /// Extend a Point3 into a Point4.
    pub fn from_point3(p: Point3<T>, w: T) -> Self {
        let Point([x, y, z]) = p;
        Point([x, y, z, w])
    }
}

/// Constructs a new point from an array. Necessary to help the compiler. Prefer
/// calling the macro `point!`, which calls `new_point` internally.
#[inline]
#[doc(hidden)]
pub fn new_point<T, const N: usize>(elements: [T; N]) -> Point<T, N> {
    Point(elements)
}

/// Construct a new [Point] of any size.
///
/// ```
/// # use al_jabr::*;
/// let p: Point<u32, 0> = point![];
/// let p = point![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let p = point![true, false, false, true];
/// ```
#[macro_export]
macro_rules! point {
    ( $($elem:expr),* $(,)? ) => {
        $crate::new_point([
            $($elem),*
        ])
    }
}

impl<T, const N: usize> Clone for Point<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Point::<T, N>(self.0.clone())
    }
}

impl<T, const N: usize> Copy for Point<T, N> where T: Copy {}

impl<T, const N: usize> From<Point<T, N>> for [T; N] {
    fn from(p: Point<T, N>) -> [T; N] {
        p.0
    }
}

impl<T, const N: usize> fmt::Debug for Point<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match N {
            0 => unimplemented!(),
            1 => write!(f, "Point {{ x: {:?} }}", self.0[0]),
            2 => write!(f, "Point {{ x: {:?}, y: {:?} }}", self.0[0], self.0[1]),
            3 => write!(
                f,
                "Point {{ x: {:?}, y: {:?}, z: {:?} }}",
                self.0[0], self.0[1], self.0[2]
            ),
            4 => write!(
                f,
                "Point {{ x: {:?}, y: {:?}, z: {:?}, w: {:?} }}",
                self.0[0], self.0[1], self.0[2], self.0[3]
            ),
            _ => write!(
                f,
                "Point {{ x: {:?}, y: {:?}, z: {:?}, w: {:?}, [..]: {:?} }}",
                self.0[0],
                self.0[1],
                self.0[2],
                self.0[3],
                &self.0[4..]
            ),
        }
    }
}

impl<T, const N: usize> Deref for Point<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for Point<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const N: usize> Hash for Point<T, N>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in 0..N {
            self.0[i].hash(state);
        }
    }
}

impl<A, B, RHS, const N: usize> PartialEq<RHS> for Point<A, N>
where
    RHS: Deref<Target = [B; N]>,
    A: PartialEq<B>,
{
    fn eq(&self, other: &RHS) -> bool {
        self.0
            .iter()
            .zip(other.deref().iter())
            .all(|(a, b)| a.eq(b))
    }
}

impl<T, const N: usize> Eq for Point<T, N> where T: Eq {}

impl<A, B, const N: usize> Add<ColumnVector<B, N>> for Point<A, N>
where
    A: Add<B>,
{
    type Output = Point<<A as Add<B>>::Output, N>;

    fn add(self, rhs: ColumnVector<B, N>) -> Self::Output {
        let lhs = Matrix([self.0]);
        let Matrix([res]) = lhs + rhs;
        Point(res)
    }
}

impl<A, B, const N: usize> Sub<ColumnVector<B, N>> for Point<A, N>
where
    A: Sub<B>,
{
    type Output = Point<<A as Sub<B>>::Output, N>;

    fn sub(self, rhs: ColumnVector<B, N>) -> Self::Output {
        let lhs = Matrix([self.0]);
        let Matrix([res]) = lhs - rhs;
        Point(res)
    }
}

impl<A, B, const N: usize> Sub<Point<B, N>> for Point<A, N>
where
    A: Sub<B>,
{
    type Output = ColumnVector<<A as Sub<B>>::Output, N>;

    fn sub(self, rhs: Point<B, N>) -> Self::Output {
        let lhs = Matrix([self.0]);
        let rhs = Matrix([rhs.0]);
        lhs - rhs
    }
}

impl<T, const N: usize> IntoIterator for Point<T, N> {
    type Item = T;
    type IntoIter = ArrayIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let Point(array) = self;
        ArrayIter {
            array: MaybeUninit::new(array),
            pos: 0,
        }
    }
}

#[cfg(feature = "rand")]
impl<T, const N: usize> Distribution<Point<T, N>> for Standard
where
    Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point<T, N> {
        let mut rand = MaybeUninit::<Point<T, N>>::uninit();
        let randp = &mut rand as *mut MaybeUninit<Point<T, N>> as *mut T;

        for i in 0..N {
            unsafe { randp.add(i).write(self.sample(rng)) }
        }

        unsafe { rand.assume_init() }
    }
}

#[cfg(feature = "serde")]
impl<T, const N: usize> Serialize for Point<T, N>
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
impl<'de, T, const N: usize> Deserialize<'de> for Point<T, N>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_tuple(N, ArrayVisitor::<[T; N]>::new())
            .map(Point)
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> From<Point<T, 2>> for mint::Point2<T> {
    fn from(p: Point<T, 2>) -> mint::Point2<T> {
        mint::Point2 {
            x: p.0[0],
            y: p.0[1],
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Point2<T>> for Point<T, 2> {
    fn from(mint_point: mint::Point2<T>) -> Self {
        Point([mint_point.x, mint_point.y])
    }
}

#[cfg(feature = "mint")]
impl<T: Copy> From<Point<T, 3>> for mint::Point3<T> {
    fn from(p: Point<T, 3>) -> mint::Point3<T> {
        mint::Point3 {
            x: p.0[0],
            y: p.0[1],
            z: p.0[2],
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Point3<T>> for Point<T, 3> {
    fn from(mint_point: mint::Point3<T>) -> Self {
        Point([mint_point.x, mint_point.y, mint_point.z])
    }
}

#[cfg(any(feature = "approx", test))]
use approx::AbsDiffEq;

#[cfg(any(feature = "approx", test))]
impl<T: AbsDiffEq, const N: usize> AbsDiffEq for Point<T, N>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        for i in 0..N {
            if !T::abs_diff_eq(&self.0[i], &other[i], epsilon) {
                return false;
            }
        }
        true
    }
}

#[cfg(feature = "approx")]
use approx::{RelativeEq, UlpsEq};

#[cfg(feature = "approx")]
impl<T, const N: usize> RelativeEq for Point<T, N>
where
    T: RelativeEq,
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        for i in 0..N {
            if !T::relative_eq(&self[i], &other[i], epsilon, max_relative) {
                return false;
            }
        }
        true
    }
}

#[cfg(feature = "approx")]
impl<T, const N: usize> UlpsEq for Point<T, N>
where
    T: UlpsEq,
    T::Epsilon: Copy,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        for i in 0..N {
            if !T::ulps_eq(&self[i], &other[i], epsilon, max_ulps) {
                return false;
            }
        }
        true
    }
}

*/
