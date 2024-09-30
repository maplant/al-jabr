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
//! Points can be freely converted to and from vectors via `from_vec`
//! and `to_vec`.

use super::*;

/// A point in 2-dimensional space.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

impl<T> From<[T; 2]> for Point2<T> {
    fn from(arr: [T; 2]) -> Self {
        let [x, y] = arr;
        Self { x, y }
    }
}

impl<T> Point2<T> {
    /// Construct a new point
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    /// Convert a point from a [Vector2]
    pub fn from_vec(vec: Vector2<T>) -> Self {
        Self { x: vec.x, y: vec.y }
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
        Self { x, y }
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
}

impl<T> IntoIterator for Point2<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, 2>;

    fn into_iter(self) -> Self::IntoIter {
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

impl<T> Index<usize> for Point2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Out of range"),
        }
    }
}

impl<T> IndexMut<usize> for Point2<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Out of range"),
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

impl<A, B> AddAssign<Vector2<B>> for Point2<A>
where
    A: AddAssign<B>,
{
    fn add_assign(&mut self, rhs: Vector2<B>) {
        self.x += rhs.x;
        self.y += rhs.y;
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

impl<A, B> SubAssign<Vector2<B>> for Point2<A>
where
    A: SubAssign<B>,
{
    fn sub_assign(&mut self, rhs: Vector2<B>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
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
        T::abs_diff_eq(&self.x, &other.x, epsilon) && T::abs_diff_eq(&self.y, &other.y, epsilon)
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
        T::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && T::relative_eq(&self.y, &other.y, epsilon, max_relative)
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
        T::ulps_eq(&self.x, &other.x, epsilon, max_ulps)
            && T::ulps_eq(&self.y, &other.y, epsilon, max_ulps)
    }
}

#[cfg(feature = "rand")]
impl<T> Distribution<Point2<T>> for Standard
where
    Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point2<T> {
        Point2 {
            x: self.sample(rng),
            y: self.sample(rng),
        }
    }
}

/// A point in 3-dimensional space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

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
        Self { x, y, z }
    }

    pub fn to_col(self) -> ColumnVector<T, 3> {
        Matrix([[self.x, self.y, self.z]])
    }

    /// Construct a new point by mapping the components of the old point.
    pub fn map<B>(self, mut f: impl FnMut(T) -> B) -> Point3<B> {
        Point3 {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
        }
    }

    /// Construct a point where each element is the pair of the components
    /// of the two points.
    pub fn zip<B>(self, p2: Point3<B>) -> Point3<(T, B)> {
        Point3 {
            x: (self.x, p2.x),
            y: (self.y, p2.y),
            z: (self.z, p2.z),
        }
    }

    /// Iterate over the point, in x, y, z order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.x, &self.y, &self.z].into_iter()
    }

    /// Mutably iterate over the point, in x, y, z order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        [&mut self.x, &mut self.y, &mut self.z].into_iter()
    }
}

impl<T> IntoIterator for Point3<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, 3>;

    fn into_iter(self) -> Self::IntoIter {
        [self.x, self.y, self.z].into_iter()
    }
}

impl<T> From<[T; 3]> for Point3<T> {
    fn from(arr: [T; 3]) -> Self {
        let [x, y, z] = arr;
        Self { x, y, z }
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

impl<T> Index<usize> for Point3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Out of range"),
        }
    }
}

impl<T> IndexMut<usize> for Point3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Out of range"),
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

impl<A, B> AddAssign<Vector3<B>> for Point3<A>
where
    A: AddAssign<B>,
{
    fn add_assign(&mut self, rhs: Vector3<B>) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
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

impl<A, B> SubAssign<Vector3<B>> for Point3<A>
where
    A: SubAssign<B>,
{
    fn sub_assign(&mut self, rhs: Vector3<B>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
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
        T::abs_diff_eq(&self.x, &other.x, epsilon)
            && T::abs_diff_eq(&self.y, &other.y, epsilon)
            && T::abs_diff_eq(&self.z, &other.z, epsilon)
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
        T::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && T::relative_eq(&self.y, &other.y, epsilon, max_relative)
            && T::relative_eq(&self.z, &other.z, epsilon, max_relative)
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
        T::ulps_eq(&self.x, &other.x, epsilon, max_ulps)
            && T::ulps_eq(&self.y, &other.y, epsilon, max_ulps)
            && T::ulps_eq(&self.z, &other.z, epsilon, max_ulps)
    }
}

#[cfg(feature = "rand")]
impl<T> Distribution<Point3<T>> for Standard
where
    Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point3<T> {
        Point3 {
            x: self.sample(rng),
            y: self.sample(rng),
            z: self.sample(rng),
        }
    }
}

/// A point in 4-dimensional space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Point4<T> {
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }

    pub fn from_vec(vec: Vector4<T>) -> Self {
        Self {
            x: vec.x,
            y: vec.y,
            z: vec.z,
            w: vec.w,
        }
    }

    pub fn to_vec(self) -> Vector4<T> {
        Vector4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w: self.w,
        }
    }

    pub fn from_col(Matrix([[x, y, z, w]]): ColumnVector<T, 4>) -> Self {
        Self { x, y, z, w }
    }

    pub fn to_col(self) -> ColumnVector<T, 4> {
        Matrix([[self.x, self.y, self.z, self.w]])
    }

    /// Construct a new point by mapping the components of the old point.
    pub fn map<B>(self, mut f: impl FnMut(T) -> B) -> Point4<B> {
        Point4 {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
            w: f(self.w),
        }
    }

    /// Construct a point where each element is the pair of the components
    /// of the two points.
    pub fn zip<B>(self, p2: Point4<B>) -> Point4<(T, B)> {
        Point4 {
            x: (self.x, p2.x),
            y: (self.y, p2.y),
            z: (self.z, p2.z),
            w: (self.w, p2.w),
        }
    }

    /// Iterate over the point, in x, y, z order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.x, &self.y, &self.z, &self.w].into_iter()
    }

    /// Mutably iterate over the point, in x, y, z order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        [&mut self.x, &mut self.y, &mut self.z, &mut self.w].into_iter()
    }
}

impl<T> IntoIterator for Point4<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, 4>;

    fn into_iter(self) -> Self::IntoIter {
        [self.x, self.y, self.z, self.w].into_iter()
    }
}

impl<T> From<[T; 4]> for Point4<T> {
    fn from(arr: [T; 4]) -> Self {
        let [x, y, z, w] = arr;
        Self { x, y, z, w }
    }
}

impl<T: Zero> Point4<T> {
    pub fn origin() -> Self {
        Self {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
            w: T::zero(),
        }
    }
}

impl<T> Index<usize> for Point4<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Out of range"),
        }
    }
}

impl<T> IndexMut<usize> for Point4<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Out of range"),
        }
    }
}

impl<A, B> Add<Vector4<B>> for Point4<A>
where
    A: Add<B>,
{
    type Output = Point4<A::Output>;

    fn add(self, rhs: Vector4<B>) -> Self::Output {
        Point4 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl<A, B> AddAssign<Vector4<B>> for Point4<A>
where
    A: AddAssign<B>,
{
    fn add_assign(&mut self, rhs: Vector4<B>) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl<A, B> Sub<Vector4<B>> for Point4<A>
where
    A: Sub<B>,
{
    type Output = Point4<A::Output>;

    fn sub(self, rhs: Vector4<B>) -> Self::Output {
        Point4 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl<A, B> SubAssign<Vector4<B>> for Point4<A>
where
    A: SubAssign<B>,
{
    fn sub_assign(&mut self, rhs: Vector4<B>) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl<A, B> Sub<Point4<B>> for Point4<A>
where
    A: Sub<B>,
{
    type Output = Vector4<A::Output>;

    fn sub(self, rhs: Point4<B>) -> Self::Output {
        Vector4 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

#[cfg(any(feature = "approx", test))]
impl<T: AbsDiffEq> AbsDiffEq for Point4<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        T::abs_diff_eq(&self.x, &other.x, epsilon)
            && T::abs_diff_eq(&self.y, &other.y, epsilon)
            && T::abs_diff_eq(&self.z, &other.z, epsilon)
            && T::abs_diff_eq(&self.w, &other.w, epsilon)
    }
}

#[cfg(feature = "approx")]
impl<T> RelativeEq for Point4<T>
where
    T: RelativeEq,
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        T::relative_eq(&self.x, &other.x, epsilon, max_relative)
            && T::relative_eq(&self.y, &other.y, epsilon, max_relative)
            && T::relative_eq(&self.z, &other.z, epsilon, max_relative)
            && T::relative_eq(&self.w, &other.w, epsilon, max_relative)
    }
}

#[cfg(feature = "approx")]
impl<T> UlpsEq for Point4<T>
where
    T: UlpsEq,
    T::Epsilon: Copy,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        T::ulps_eq(&self.x, &other.x, epsilon, max_ulps)
            && T::ulps_eq(&self.y, &other.y, epsilon, max_ulps)
            && T::ulps_eq(&self.z, &other.z, epsilon, max_ulps)
            && T::ulps_eq(&self.w, &other.w, epsilon, max_ulps)
    }
}

#[cfg(feature = "rand")]
impl<T> Distribution<Point4<T>> for Standard
where
    Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point4<T> {
        Point4 {
            x: self.sample(rng),
            y: self.sample(rng),
            z: self.sample(rng),
            w: self.sample(rng),
        }
    }
}
