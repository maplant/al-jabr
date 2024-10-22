//! Convenient structs for common Vector dimensions (1, 2, 3, and 4)

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

macro_rules! implement_vector {
    ( $name:ident, $size:literal, x $(, $field:ident )* ) => {
        impl<T> Zero for $name<T>
        where
            T: Zero,
        {
            fn zero() -> Self {
                Self {
                    x: T::zero(),
                    $( $field: T::zero(), )*
                }
            }

            /// Returns true if the value is the additive identity.
            fn is_zero(&self) -> bool {
                self.x.is_zero() $( && self.$field.is_zero() )*
            }
        }

        impl<T> Default for $name<T>
        where
            T: Zero,
        {
            fn default() -> Self {
                Self::zero()
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
            fn from(vec: $name<T>) -> Self {
                [
                    vec.x,
                    $( vec.$field, )*
                ]
            }
        }

        impl<T> $name<T> {
            /// Iterate over the components of the vector
            pub fn iter(&self) -> impl Iterator<Item = &T>  {
                [ &self.x, $( &self.$field ),* ].into_iter()
            }

            /// Mutably iterate over the components of the vector
            pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T>  {
                [ &mut self.x, $( &mut self.$field ),* ].into_iter()
            }
        }

        impl<T> IntoIterator for $name<T> {
            type Item = T;
            type IntoIter = std::array::IntoIter<T, $size>;

            fn into_iter(self) -> Self::IntoIter {
                 [ self.x, $( self.$field ),* ].into_iter()
            }
        }

        impl<T> $name<T> {
            /// Construct a new Vector
            pub const fn new(x: T, $($field: T,)*) -> Self {
                Self { x, $($field,)* }
            }

            /// Transpose the vector into a [ColumnVector]
            pub fn transpose(self) -> ColumnVector<T, $size> {
                Matrix([[self.x, $(self.$field,)*]])
            }

            /// Construct a new vector by mapping the components of the old vector.
            pub fn map<O>(self, mut f: impl FnMut(T) -> O) -> $name<O> {
                $name {
                    x: f(self.x),
                    $( $field: f(self.$field), )*
                }
            }

            /// Construct a vector where each element is the pair of the components
            /// of the two vectors.
            pub fn zip<B>(self, v2: $name<B>) -> $name<(T, B)> {
                $name {
                    x: (self.x, v2.x),
                    $( $field: (self.$field, v2.$field), )*
                }
            }
        }

        impl<T> $name<T>
        where
            T: PartialOrd + Clone,
        {
            /// Return the largest value found in the vector, along with the associated index. If there is
            /// no largest value returns the x component.
            #[allow(unused_mut)]
            pub fn argmax(&self) -> (usize, T) {
                let mut i_max = 0;
                let mut v_max = self.x.clone();
                $(
                    if self.$field > v_max {
                        i_max = field_to_index!($field);
                        v_max = self.$field.clone();
                    }
                )*
                (i_max, v_max)
            }

            /// Return the largest value in the vector. If there is no largest value, returns the x component.
            #[allow(unused_mut)]
            pub fn max(&self) -> T {
                let mut v_max = self.x.clone();
                $(
                    if self.$field > v_max {
                        v_max = self.$field.clone();
                    }
                )*
                v_max
            }

            /// Return the smallest value found in the vector, along with the associated index. If there is
            /// no smallest value returns the x component.
            #[allow(unused_mut)]
            pub fn argmin(&self) -> (usize, T) {
                let mut i_max = 0;
                let mut v_max = self.x.clone();
                $(
                    if self.$field > v_max {
                        i_max = field_to_index!($field);
                        v_max = self.$field.clone();
                    }
                )*
                (i_max, v_max)
            }

            /// Return the smallest value in the vector. If there is no smallest value, returns the x
            /// component.
            #[allow(unused_mut)]
            pub fn min(&self) -> T {
                let mut v_max = self.x.clone();
                $(
                    if self.$field > v_max {
                        v_max = self.$field.clone();
                    }
                )*
                v_max
            }
        }

        impl<T> Index<usize> for $name<T> {
            type Output = T;

            fn index(&self, index: usize) -> &T {
                match index {
                    0 => &self.x,
                    $( field_to_index!($field) => &self.$field, )*
                        _ => panic!("Out of range"),
                }
            }
        }

        impl<T> IndexMut<usize> for $name<T> {
            fn index_mut(&mut self, index: usize) -> &mut T {
                match index {
                    0 => &mut self.x,
                    $( field_to_index!($field) => &mut self.$field, )*
                        _ => panic!("Out of range"),
                }
            }
        }

        impl<A> Neg for $name<A>
        where
            A: Neg,
        {
            type Output = $name<<A as Neg>::Output>;

            fn neg(self) -> Self::Output {
                $name {
                    x: -self.x,
                    $( $field: -self.$field, )*
                }
            }
        }

        impl<A, B> Add<$name<B>> for $name<A>
        where
            A: Add<B>
        {
            type Output = $name<A::Output>;

            fn add(self, rhs: $name<B>) -> $name<A::Output> {
                $name {
                    x: self.x + rhs.x,
                    $( $field: self.$field + rhs.$field, )*
                }
            }
        }

        impl<A, B> AddAssign<$name<B>> for $name<A>
        where
            A: AddAssign<B>
        {
            fn add_assign(&mut self, rhs: $name<B>) {
                self.x += rhs.x;
                $( self.$field += rhs.$field; )*
            }
        }

        impl<A, B> Sub<$name<B>> for $name<A>
        where
            A: Sub<B>
        {
            type Output = $name<A::Output>;

            fn sub(self, rhs: $name<B>) -> $name<A::Output> {
                $name {
                    x: self.x - rhs.x,
                    $( $field: self.$field - rhs.$field, )*
                }
            }
        }

        impl<A, B> SubAssign<$name<B>> for $name<A>
        where
            A: SubAssign<B>
        {
            fn sub_assign(&mut self, rhs: $name<B>) {
                self.x -= rhs.x;
                $( self.$field -= rhs.$field; )*
            }
        }

        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<A> Mul<SquareMatrix<A, $size>> for $name<A>
        where
            A: Mul,
            SquareMatrix<A, $size>: Mul<Matrix<A, $size, 1>, Output = ColumnVector<A::Output, $size>>,
        {
            type Output = $name<A::Output>;

            fn mul(self, rhs: SquareMatrix<A, $size>) -> Self::Output {
                let Matrix([[x, $($field),*]]) = rhs * self.transpose();
                $name { x, $($field),* }
            }
        }

        impl<A> Mul<A> for $name<A>
        where
            A: Mul + Clone,
        {
            type Output = $name<A::Output>;

            fn mul(self, rhs: A) -> $name<A::Output> {
                $name {
                    $( $field: self.$field * rhs.clone(), )*
                        x: self.x * rhs
                }
            }
        }

        impl<A, B> MulAssign<B> for $name<A>
        where
            A: MulAssign<B>,
            B: Clone,
        {
            fn mul_assign(&mut self, rhs: B) {
                $( self.$field *= rhs.clone(); )*
                    self.x *= rhs;
            }
        }

        impl<A, B> Div<B> for $name<A>
        where
            A: Div<B>,
            B: Clone,
        {
            type Output = $name<A::Output>;

            fn div(self, rhs: B) -> $name<A::Output> {
                $name {
                    $( $field: self.$field / rhs.clone(), )*
                    x: self.x / rhs
                }
            }
        }

        impl<A, B> DivAssign<B> for $name<A>
        where
            A: DivAssign<B>,
            B: Clone,
        {
            fn div_assign(&mut self, rhs: B) {
                $( self.$field /= rhs.clone(); )*
                    self.x /= rhs;
            }
        }

        impl<T> VectorSpace for $name<T>
        where
            T: Clone + Zero,
            T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>,
        {
            type Scalar = T;
        }

        impl<T> MetricSpace for $name<T>
        where
            T: Clone + Zero,
            T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>,
        {
            type Metric = T;

            fn distance2(self, other: Self) -> T {
                (other - self).magnitude2()
            }
        }

        impl<T> InnerSpace for $name<T>
        where
            T: Clone + Zero,
            T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>,
        {
            fn dot(self, rhs: Self) -> T {
                self.x * rhs.x $( + self.$field * rhs.$field )*
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
                self.x.abs_diff_eq(&other.x, epsilon)
                    $( && self.$field.abs_diff_eq(&other.$field, epsilon) )*
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
                self.x.relative_eq(&other.x, epsilon, max_relative)
                    $( && self.$field.relative_eq(&other.$field, epsilon, max_relative) )*
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
                self.x.ulps_eq(&other.x, epsilon, max_ulps)
                    $( && self.$field.ulps_eq(&other.$field, epsilon, max_ulps) )*
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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
/// A vector in 1-dimensional space.
pub struct Vector1<S> {
    /// The x component of the vector.
    pub x: S,
}

impl<T> Vector1<T> {
    /// Extend the vector into a [Vector2].
    pub fn extend(self, y: T) -> Vector2<T> {
        Vector2::new(self.x, y)
    }
}

implement_vector!(Vector1, 1, x);

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
/// A vector in 2-dimensional space.
pub struct Vector2<S> {
    /// The x component of the vector.
    pub x: S,
    /// The y component of the vector.
    pub y: S,
}

impl<T> Vector2<T> {
    /// Drop the y component of the vector, creating a [Vector1].
    pub fn truncate(self) -> Vector1<T> {
        Vector1::new(self.x)
    }

    /// Extend the vector into a [Vector3].
    pub fn extend(self, z: T) -> Vector3<T> {
        Vector3::new(self.x, self.y, z)
    }
}

impl<T> Vector2<T>
where
    T: One + Zero,
{
    /// Construct a normal 4-dimensional vector in the X direction.
    pub fn unit_x() -> Self {
        Self::new(T::one(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Y direction.
    pub fn unit_y() -> Self {
        Self::new(T::zero(), T::one())
    }
}

implement_vector!(Vector2, 2, x, y);

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
/// A vector in 3-dimensional space.
pub struct Vector3<S> {
    /// The x component of the vector.
    pub x: S,
    /// The y component of the vector.
    pub y: S,
    /// The z component of the vector.
    pub z: S,
}

impl<T> Vector3<T> {
    /// Drop the z component of the vector, creating a [Vector2].
    pub fn truncate(self) -> Vector2<T> {
        Vector2::new(self.x, self.y)
    }

    /// Extend the vector into a [Vector4].
    pub fn extend(self, w: T) -> Vector4<T> {
        Vector4::new(self.x, self.y, self.z, w)
    }
}

impl<T> Vector3<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Clone,
{
    pub fn cross(self, rhs: Vector3<T>) -> Self {
        let Vector3 {
            x: x0,
            y: y0,
            z: z0,
        } = self;
        let Vector3 {
            x: x1,
            y: y1,
            z: z1,
        } = rhs;
        Self {
            x: y0.clone() * z1.clone() - z0.clone() * y1.clone(),
            y: z0 * x1.clone() - x0.clone() * z1,
            z: x0 * y1 - y0 * x1,
        }
    }
}

impl<T> Vector3<T>
where
    T: One + Zero,
{
    /// Construct a normal 4-dimensional vector in the X direction.
    pub fn unit_x() -> Self {
        Self::new(T::one(), T::zero(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Y direction.
    pub fn unit_y() -> Self {
        Self::new(T::zero(), T::one(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Z direction.
    pub fn unit_z() -> Self {
        Self::new(T::zero(), T::zero(), T::one())
    }
}

implement_vector!(Vector3, 3, x, y, z);

macro_rules! swizzle3 {
    ($a:ident, $x:ident, $y:ident, $z:ident) => {
        swizzle3!{ second, $a, $x, $x, $y, $z }
        swizzle3!{ second, $a, $y, $x, $y, $z }
        swizzle3!{ second, $a, $z, $x, $y, $z }
    };

    ( second, $a:ident, $b:ident, $x:ident, $y:ident, $z:ident) => {
        paste::item! {
            #[doc(hidden)]
            pub fn [< $a $b >](&self) -> Vector2<T> {
                Vector2::new(
                    self.$a.clone(),
                    self.$b.clone(),
                )
            }
        }

        swizzle3!{ third, $a, $b, $x, $x, $y, $z }
        swizzle3!{ third, $a, $b, $y, $x, $y, $z }
        swizzle3!{ third, $a, $b, $z, $x, $y, $z }
    };

    ( third, $a:ident, $b:ident, $c:ident, $x:ident, $y:ident, $z:ident) => {
        paste::item! {
            #[doc(hidden)]
            pub fn [< $a $b $c >](&self) -> Vector3<T> {
                Vector3::new(
                    self.$a.clone(),
                    self.$b.clone(),
                    self.$c.clone(),
                )
            }
        }

        swizzle3!{ fourth, $a, $b, $c, $x }
        swizzle3!{ fourth, $a, $b, $c, $y }
        swizzle3!{ fourth, $a, $b, $c, $z }
    };

    ( fourth, $a:ident, $b:ident, $c:ident, $d:ident) => {
        paste::item! {
            #[doc(hidden)]
            pub fn [< $a $b $c $d >](&self) -> Vector4<T> {
                Vector4::new(
                    self.$a.clone(),
                    self.$b.clone(),
                    self.$c.clone(),
                    self.$d.clone(),
                )
            }
        }
    };
}

impl<T> Vector3<T>
where
    T: Clone,
{
    swizzle3! {x, x, y, z}
    swizzle3! {y, x, y, z}
    swizzle3! {z, x, y, z}
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
/// A vector in 4-dimensional space.
pub struct Vector4<S> {
    /// The x component of the vector.
    pub x: S,
    /// The y component of the vector.
    pub y: S,
    /// The z component of the vector.
    pub z: S,
    /// The w component of the vector.
    pub w: S,
}

impl<T> Vector4<T> {
    /// Drop the w component of the vector, creating a [Vector3].
    pub fn truncate(self) -> Vector3<T> {
        Vector3::new(self.x, self.y, self.z)
    }
}

impl<T> Vector4<T>
where
    T: One + Zero,
{
    /// Construct a normal 4-dimensional vector in the X direction.
    pub fn unit_x() -> Self {
        Self::new(T::one(), T::zero(), T::zero(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Y direction.
    pub fn unit_y() -> Self {
        Self::new(T::zero(), T::one(), T::zero(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Z direction.
    pub fn unit_z() -> Self {
        Self::new(T::zero(), T::zero(), T::one(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the W direction.
    pub fn unit_w() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::one())
    }
}

implement_vector!(Vector4, 4, x, y, z, w);

macro_rules! swizzle4 {
    ($a:ident, $x:ident, $y:ident, $z:ident, $w:ident) => {
        swizzle4!{ $a, $x, $x, $y, $z, $w }
        swizzle4!{ $a, $y, $x, $y, $z, $w }
        swizzle4!{ $a, $z, $x, $y, $z, $w }
        swizzle4!{ $a, $w, $x, $y, $z, $w }
    };

    ($a:ident, $b:ident, $x:ident, $y:ident, $z:ident, $w:ident) => {
        paste::item! {
            #[doc(hidden)]
            pub fn [< $a $b >](&self) -> Vector2<T> {
                Vector2::new(
                    self.$a.clone(),
                    self.$b.clone(),
                )
            }
        }

        swizzle4!{ $a, $b, $x, $x, $y, $z, $w }
        swizzle4!{ $a, $b, $y, $x, $y, $z, $w }
        swizzle4!{ $a, $b, $z, $x, $y, $z, $w }
        swizzle4!{ $a, $b, $w, $x, $y, $z, $w }
    };

    ($a:ident, $b:ident, $c:ident, $x:ident, $y:ident, $z:ident, $w:ident) => {
        paste::item! {
            #[doc(hidden)]
            pub fn [< $a $b $c >](&self) -> Vector3<T> {
                Vector3::new(
                    self.$a.clone(),
                    self.$b.clone(),
                    self.$c.clone(),
                )
            }
        }

        swizzle4!{ $a, $b, $c, $x }
        swizzle4!{ $a, $b, $c, $y }
        swizzle4!{ $a, $b, $c, $z }
        swizzle4!{ $a, $b, $c, $w }
    };

    ($a:ident, $b:ident, $c:ident, $d:ident) => {
        paste::item! {
            #[doc(hidden)]
            pub fn [< $a $b $c $d >](&self) -> Vector4<T> {
                Vector4::new(
                    self.$a.clone(),
                    self.$b.clone(),
                    self.$c.clone(),
                    self.$d.clone(),
                )
            }
        }
    };
}

impl<T> Vector4<T>
where
    T: Clone,
{
    swizzle4! {x, x, y, z, w}
    swizzle4! {y, x, y, z, w}
    swizzle4! {z, x, y, z, w}
    swizzle4! {w, x, y, z, w}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(Vector1::<u32>::zero(), Vector1::new(0));
        assert_eq!(Vector2::<u32>::zero(), Vector2::new(0, 0));
        assert_eq!(Vector3::<u32>::zero(), Vector3::new(0, 0, 0));
        assert_eq!(Vector4::<u32>::zero(), Vector4::new(0, 0, 0, 0));
    }

    #[test]
    fn test_index() {
        let a = Vector1::new(0_u32);
        assert_eq!(a.x, 0_u32);
        let b = Vector2::new(1_u32, 2);
        assert_eq!(b.x, 1);
        assert_eq!(b.y, 2);
    }

    #[test]
    fn test_eq() {
        let a = Vector3::new(0_u32, 1, 2);
        let b = Vector3::new(1_u32, 2, 3);
        let c = Vector3::new(0_u32, 1, 2);
        assert_ne!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn test_addition() {
        let a = Vector1::new(0_u32);
        let b = Vector1::new(1_u32);
        let c = Vector1::new(2_u32);
        assert_eq!(a + b, b);
        assert_eq!(b + b, c);
        let a = Vector2::new(0_u32, 1);
        let b = Vector2::new(1_u32, 2);
        let c = Vector2::new(1_u32, 3);
        let d = Vector2::new(2_u32, 5);
        assert_eq!(a + b, c);
        assert_eq!(b + c, d);
        let mut c = Vector2::new(1_u32, 3);
        let d = Vector2::new(2_u32, 5);
        c += d;
        let e = Vector2::new(3_u32, 8);
        assert_eq!(c, e);
    }

    #[test]
    fn test_subtraction() {
        let mut a = Vector2::new(3_u32, 4);
        let b = Vector2::new(1, 2);
        let c = Vector2::new(2, 2);
        assert_eq!(a - c, b);
        a -= b;
        assert_eq!(a, c);
    }

    #[test]
    fn test_negation() {
        let a = Vector4::new(1_i32, 2, 3, 4);
        let b = Vector4::new(-1_i32, -2, -3, -4);
        assert_eq!(-a, b);
    }

    #[test]
    fn test_scale() {
        let a = Vector4::new(2.0_f32, 4.0, 2.0, 4.0);
        let b = Vector4::new(4.0_f32, 8.0, 4.0, 8.0);
        let c = Vector4::new(1.0_f32, 2.0, 1.0, 2.0);
        assert_eq!(a * 2.0, b);
        assert_eq!(a / 2.0, c);
    }

    #[test]
    fn test_cross() {
        let a = Vector3::new(1isize, 2isize, 3isize);
        let b = Vector3::new(4isize, 5isize, 6isize);
        let r = Vector3::new(-3isize, 6isize, -3isize);
        assert_eq!(a.cross(b), r);
    }

    #[test]
    fn test_distance() {
        let a = Vector1::new(0.0_f32);
        let b = Vector1::new(1.0_f32);
        assert_eq!(a.distance2(b), 1.0);
        let a = Vector1::new(0.0_f32);
        let b = Vector1::new(2.0_f32);
        assert_eq!(a.distance2(b), 4.0);
        assert_eq!(a.distance(b), 2.0);
        let a = Vector2::new(0.0_f32, 0.0);
        let b = Vector2::new(1.0_f32, 1.0);
        assert_eq!(a.distance2(b), 2.0);

        let a = Vector2::new(1i32, 1);
        let b = Vector2::new(5i32, 5);
        assert_eq!(a.distance2(b), 32); // distance method not implemented.
        assert_eq!((b - a).magnitude2(), 32);
    }

    #[test]
    fn test_normalize() {
        let a = Vector1::new(5.0);
        assert_eq!(a.magnitude(), 5.0);
        let a_norm = a.normalize();
        assert_eq!(a_norm, Vector1::new(1.0));
    }
}
