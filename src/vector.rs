//! Convenient structs for common Vector dimensions (2, 3, and 4)

use super::*;

/*
macro_rules! apply {
    ( $s:ident, $fun:ident, $x:ident ) => {
        $s.$x
    };

    ( $s:ident, $fun:ident, $x:ident, $( $field:ident ),+ ) => {
        $fun( $s.$x, apply!( $s, $fun, $( $field ),+ ) )
    }
}
*/

macro_rules! count {
    ( $x:ident ) => {
        1
    };

    ( $x1:ident, $( $xn:ident ),* ) => {
        1 + count!($( $xn ),*)
    }
}

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
    ( $name:ident, x, $( $field:ident ),+ ) => {
        impl<T> Zero for $name<T>
        where
            T: Zero,
        {
            fn zero() -> Self {
                Self {
                    x: T::zero(),
                    $( $field: T::zero(), )+
                }
            }

            /// Returns true if the value is the additive identity.
            fn is_zero(&self) -> bool {
                self.x.is_zero() $( && self.$field.is_zero() )+
            }
        }

        impl<T> $name<T> {
            /// Iterate over the components of the vector
            pub fn iter(&self) -> impl Iterator<Item = &T>  {
                [ &self.x, $( &self.$field ),+ ].into_iter()
            }

            /// Mutably iterate over the components of the vector
            pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T>  {
                [ &mut self.x, $( &mut self.$field ),+ ].into_iter()
            }
        }

        impl<T> IntoIterator for $name<T> {
            type Item = T;
            type IntoIter = std::array::IntoIter<T, { count!(x, $( $field ),+) }>;

            fn into_iter(self) -> Self::IntoIter {
                 [ self.x, $( self.$field ),+ ].into_iter()
            }
        }

        impl<T> $name<T> {
            pub const fn new(x: T, $($field: T,)+) -> Self {
                Self { x, $($field,)+ }
            }

            /// Transpose the vector into a [ColumnVector]
            pub fn transpose(self) -> ColumnVector<T, { count!(x, $( $field ),+) }> {
                Matrix([[self.x, $(self.$field,)+]])
            }

            pub fn map<O>(self, mut f: impl FnMut(T) -> O) -> $name<O> {
                $name {
                    x: f(self.x),
                    $( $field: f(self.$field), )+
                }
            }
        }

        impl<T> $name<T>
        where
            T: PartialOrd + Clone,
        {
            /// Return the largest value found in the vector, along with the associated index. If there is
            /// no largest value returns the first value ([Self::x]).
            pub fn argmax(&self) -> (usize, T) {
                let mut i_max = 0;
                let mut v_max = self.x.clone();
                $(
                    if self.$field > v_max {
                        i_max = field_to_index!($field);
                        v_max = self.$field.clone();
                    }
                )+
                (i_max, v_max)
            }

            /// Return the largest value in the vector. If there is no largest value, returns the first value
            /// ([Self::x]).
            pub fn max(&self) -> T {
                let mut v_max = self.x.clone();
                $(
                    if self.$field > v_max {
                        v_max = self.$field.clone();
                    }
                )+
                v_max
            }

            /// Return the smallest value found in the vector, along with the associated index. If there is
            /// no smallest value returns the first value ([Self::x]).
            pub fn argmin(&self) -> (usize, T) {
                let mut i_max = 0;
                let mut v_max = self.x.clone();
                $(
                    if self.$field > v_max {
                        i_max = field_to_index!($field);
                        v_max = self.$field.clone();
                    }
                )+
                (i_max, v_max)
            }

            /// Return the smallest value in the vector. If there is no smallest value, returns the first value
            /// ([Self::x]).
            pub fn min(&self) -> T {
                let mut v_max = self.x.clone();
                $(
                    if self.$field > v_max {
                        v_max = self.$field.clone();
                    }
                )+
                v_max
            }
        }

        impl<T> Index<usize> for $name<T> {
            type Output = T;

            fn index(&self, index: usize) -> &T {
                match index {
                    0 => &self.x,
                    $( field_to_index!($field) => &self.$field, )+
                        _ => panic!("Out of range"),
                }
            }
        }

        impl<T> IndexMut<usize> for $name<T> {
            fn index_mut(&mut self, index: usize) -> &mut T {
                match index {
                    0 => &mut self.x,
                    $( field_to_index!($field) => &mut self.$field, )+
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
                    $( $field: -self.$field, )+
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
                    $( $field: self.$field + rhs.$field, )+
                }
            }
        }

        impl<A, B> AddAssign<$name<B>> for $name<A>
        where
            A: AddAssign<B>
        {
            fn add_assign(&mut self, rhs: $name<B>) {
                self.x += rhs.x;
                $( self.$field += rhs.$field; )+
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
                    $( $field: self.$field - rhs.$field, )+
                }
            }
        }

        impl<A, B> SubAssign<$name<B>> for $name<A>
        where
            A: SubAssign<B>
        {
            fn sub_assign(&mut self, rhs: $name<B>) {
                self.x -= rhs.x;
                $( self.$field -= rhs.$field; )+
            }
        }

        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<A> Mul<SquareMatrix<A, { count!(x, $($field),+) }>> for $name<A>
        where
            A: Mul,
            SquareMatrix<A, { count!(x, $($field),+) }>: Mul<Matrix<A, { count!(x, $($field),+) }, 1>, Output = ColumnVector<A::Output, { count!(x, $($field),+) }>> + Clone,
        {
            type Output = $name<A::Output>;

            fn mul(self, rhs: SquareMatrix<A, { count!(x, $($field),+) }>) -> Self::Output {
                let Matrix([[x, $($field),+]]) = rhs * self.transpose();
                $name { x, $($field),+ }
            }
        }

        impl<A> Mul<A> for $name<A>
        where
            A: Mul + Clone,
        {
            type Output = $name<A::Output>;

            fn mul(self, rhs: A) -> $name<A::Output> {
                $name {
                    $( $field: self.$field * rhs.clone(), )+
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
                $( self.$field *= rhs.clone(); )+
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
                    $( $field: self.$field / rhs.clone(), )+
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
                $( self.$field /= rhs.clone(); )+
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
                self.x * rhs.x $( + self.$field * rhs.$field )+
                // self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
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
                    $( && self.$field.abs_diff_eq(&other.$field, epsilon) )+
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
                    $( && self.$field.relative_eq(&other.$field, epsilon, max_relative) )+
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
                    $( && self.$field.ulps_eq(&other.$field, epsilon, max_ulps) )+
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
                    $( $field: self.sample(rng), )+
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

impl<T> From<[T; 2]> for Vector2<T> {
    fn from(arr: [T; 2]) -> Self {
        let [x, y] = arr;
        Self { x, y }
    }
}

impl<T> Vector2<T>
where
    T: One + Zero,
{
    /// Construct a normal 4-dimensional vector in the X direction
    pub fn unit_x() -> Self {
        Self::new(T::one(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Y direction
    pub fn unit_y() -> Self {
        Self::new(T::zero(), T::one())
    }
}

implement_vector!(Vector2, x, y);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
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

impl<T> From<[T; 3]> for Vector3<T> {
    fn from(arr: [T; 3]) -> Self {
        let [x, y, z] = arr;
        Self { x, y, z }
    }
}

impl<T> Vector3<T>
where
    T: One + Zero,
{
    /// Construct a normal 4-dimensional vector in the X direction
    pub fn unit_x() -> Self {
        Self::new(T::one(), T::zero(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Y direction
    pub fn unit_y() -> Self {
        Self::new(T::zero(), T::one(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Z direction
    pub fn unit_z() -> Self {
        Self::new(T::zero(), T::zero(), T::one())
    }
}

implement_vector!(Vector3, x, y, z);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Vector4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> From<[T; 4]> for Vector4<T> {
    fn from(arr: [T; 4]) -> Self {
        let [x, y, z, w] = arr;
        Self { x, y, z, w }
    }
}

impl<T> Vector4<T>
where
    T: One + Zero,
{
    /// Construct a normal 4-dimensional vector in the X direction
    pub fn unit_x() -> Self {
        Self::new(T::one(), T::zero(), T::zero(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Y direction
    pub fn unit_y() -> Self {
        Self::new(T::zero(), T::one(), T::zero(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the Z direction
    pub fn unit_z() -> Self {
        Self::new(T::zero(), T::zero(), T::one(), T::zero())
    }

    /// Construct a normal 4-dimensional vector in the W direction
    pub fn unit_w() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::one())
    }
}

implement_vector!(Vector4, x, y, z, w);
