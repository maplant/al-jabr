//! Convenient structs for commont Point dimensions (1, 2, 3, and 4)

use super::*;

macro_rules! field_to_index {
    ( x ) => {
        0
    };
    ( y ) => {
        1
    };
    ( z ) => {
        2
    };
    ( w ) => {
        3
    };
}

#[cfg(any(feature = "approx", test))]
use approx::AbsDiffEq;

#[cfg(feature = "approx")]
use approx::{RelativeEq, UlpsEq};

macro_rules! implement_point {
    ( $name:ident, $vec:ident, $size:literal, x $(, $field:ident )* ) => {
        impl<T> $name<T> {
            /// Construct a new Point.
            pub const fn new(x: T, $( $field: T, )*) -> Self {
                Self { x, $($field,)* }
            }

            /// Convert a Vector into a Point.
            pub fn from_vec(vec: $vec<T>) -> Self {
                Self { x: vec.x, $( $field: vec.$field, )* }
            }

            /// Convert a Point into a Vector.
            pub fn to_vec(self) -> $vec<T> {
                $vec {
                    x: self.x,
                    $( $field: self.$field, )*
                }
            }

            /// Convert a [ColumnVector] into a Point.
            pub fn from_col(Matrix([[ x, $( $field, )* ]]): ColumnVector<T, $size>) -> Self {
                Self { x, $( $field, )* }
            }

            /// Convert a Point into a [ColumnVector].
            pub fn to_col(self) -> ColumnVector<T, $size> {
                Matrix([[self.x, $( self.$field, )*]])
            }

            /// Construct a new point by mapping the components of the old point.
            pub fn map<B>(self, mut mapper: impl FnMut(T) -> B) -> $name<B> {
                $name {
                    x: mapper(self.x),
                    $( $field: mapper(self.$field), )*
                }
            }

            /// Construct a new point where each element is the pair of the components
            /// of the two arguments.
            pub fn zip<B>(self, p2: $name<B>) -> $name<(T, B)> {
                $name {
                    x: (self.x, p2.x),
                    $( $field: (self.$field, p2.$field), )*
                }
            }

            /// Iterate over the components of the Point
            pub fn iter(&self) -> impl Iterator<Item = &T>  {
                [ &self.x, $( &self.$field ),* ].into_iter()
            }

            /// Mutably iterate over the components of the Point.
            pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T>  {
                [ &mut self.x, $( &mut self.$field ),* ].into_iter()
            }
        }

        impl<T: Zero> $name<T> {
            /// Construct a point at the origin.
            pub fn origin() -> Self {
                Self {
                    x: T::zero(),
                    $( $field: T::zero(), )*
                }
            }
        }

        impl<T: Zero> Default for $name<T> {
            fn default() -> Self {
                Self::origin()
            }
        }

        impl<T> IntoIterator for $name<T> {
            type Item = T;
            type IntoIter = std::array::IntoIter<T, $size>;

            fn into_iter(self) -> Self::IntoIter {
                 [ self.x, $( self.$field ),* ].into_iter()
            }
        }

        impl<T> From<[T; $size]> for $name<T> {
            fn from(arr: [T; $size]) -> Self {
                let [ x, $( $field, )* ] = arr;
                Self {
                    x,
                    $( $field, )*
                }
            }
        }

        impl<T> From<$name<T>> for [T; $size] {
            fn from(point: $name<T>) -> Self {
                [
                    point.x,
                    $( point.$field, )*
                ]
            }
        }

        impl<T> Index<usize> for $name<T> {
            type Output = T;

            fn index(&self, index: usize) -> &T {
                match index {
                    0 => &self.x,
                    $( field_to_index!($field) => &self.$field, )*
                    _ => panic!("Index {index} out of range [0, {})", $size),
                }
            }
        }

        impl<T> IndexMut<usize> for $name<T> {
            fn index_mut(&mut self, index: usize) -> &mut T {
                match index {
                    0 => &mut self.x,
                    $( field_to_index!($field) => &mut self.$field, )*
                    _ => panic!("Index {index} out of range [0, {})", $size),
                }
            }
        }

        impl<A, B> Add<$vec<B>> for $name<A>
        where
            A: Add<B>,
        {
            type Output = $name<A::Output>;

            fn add(self, rhs: $vec<B>) -> Self::Output {
                $name {
                    x: self.x + rhs.x,
                    $( $field: self.$field + rhs.$field, )*
                }
            }
        }

        impl<A, B> AddAssign<$vec<B>> for $name<A>
        where
            A: AddAssign<B>,
        {
            fn add_assign(&mut self, rhs: $vec<B>) {
                self.x += rhs.x;
                $( self.$field += rhs.$field; )*
            }
        }

        impl<A, B> Sub<$vec<B>> for $name<A>
        where
            A: Sub<B>,
        {
            type Output = $name<A::Output>;

            fn sub(self, rhs: $vec<B>) -> Self::Output {
                $name {
                    x: self.x - rhs.x,
                    $( $field: self.$field - rhs.$field, )*
                }
            }
        }

        impl<A, B> SubAssign<$vec<B>> for $name<A>
        where
            A: SubAssign<B>,
        {
            fn sub_assign(&mut self, rhs: $vec<B>) {
                self.x -= rhs.x;
                $( self.$field -= rhs.$field; )*
            }
        }

        impl<A, B> Sub<$name<B>> for $name<A>
        where
            A: Sub<B>,
        {
            type Output = $vec<A::Output>;

            fn sub(self, rhs: $name<B>) -> Self::Output {
                $vec {
                    x: self.x - rhs.x,
                    $( $field: self.$field - rhs.$field, )*
                }
            }
        }

        #[cfg(any(feature = "approx", test))]
        impl<T: AbsDiffEq> AbsDiffEq for $name<T>
        where
            T::Epsilon: Copy,
        {
            type Epsilon = T::Epsilon;

            fn default_epsilon() -> T::Epsilon {
                T::default_epsilon()
            }

            fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
                T::abs_diff_eq(&self.x, &other.x, epsilon)
                    $( && T::abs_diff_eq(&self.$field, &other.$field, epsilon) )*
            }
        }

        #[cfg(feature = "approx")]
        impl<T> RelativeEq for $name<T>
        where
            T: RelativeEq,
            T::Epsilon: Copy,
        {
            fn default_max_relative() -> T::Epsilon {
                T::default_max_relative()
            }

            fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
                T::relative_eq(&self.x, &other.x, epsilon, max_relative)
                    $( && T::relative_eq(&self.$field, &other.$field, epsilon, max_relative) )*
            }
        }


        #[cfg(feature = "approx")]
        impl<T> UlpsEq for $name<T>
        where
            T: UlpsEq,
            T::Epsilon: Copy,
        {
            fn default_max_ulps() -> u32 {
                T::default_max_ulps()
            }

            fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
                T::ulps_eq(&self.x, &other.x, epsilon, max_ulps)
                    $( && T::ulps_eq(&self.$field, &other.$field, epsilon, max_ulps) )*
            }
        }


        #[cfg(feature = "rand")]
        impl<T> Distribution<$name<T>> for Standard
        where
            Standard: Distribution<T>,
        {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $name<T> {
                $name {
                    x: self.sample(rng),
                    $( $field: self.sample(rng), )*
                }
            }
        }
    }
}

/// A point in 1-dimensional space.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Point1<T> {
    /// x-component of the point.
    pub x: T,
}

implement_point!(Point1, Vector1, 1, x);

/// A point in 2-dimensional space.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Point2<T> {
    /// x-component of the point.
    pub x: T,
    /// y-component of the point.
    pub y: T,
}

implement_point!(Point2, Vector2, 2, x, y);

/// A point in 3-dimensional space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point3<T> {
    /// x-component of the point.
    pub x: T,
    /// y-component of the point.
    pub y: T,
    /// z-component of the point.
    pub z: T,
}

implement_point!(Point3, Vector3, 3, x, y, z);

/// A point in 4-dimensional space.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Hash, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point4<T> {
    /// x-component of the point.
    pub x: T,
    /// y-component of the point.
    pub y: T,
    /// z-component of the point.
    pub z: T,
    /// w-component of the point.
    pub w: T,
}

implement_point!(Point4, Vector4, 4, x, y, z, w);
