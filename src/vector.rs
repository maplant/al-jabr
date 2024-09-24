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
    ( x ) => { 0 };
    ( y ) => { 1 };
    ( z ) => { 2 };
    ( w ) => { 3 };
}

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


        /*
        impl<T> $name<T>
        where
            T: PartialOrd + Ord + Clone,
        {
            /// Returns the minimum value of the vector
            pub fn min(&self) -> T {
                self.iter().min().unwrap().clone()
            }

            /// Returns the maximum value of the vector
            pub fn max(&self) -> T {
                self.iter().max().unwrap().clone()
            }

            pub fn argmax(&self) -> T {
                self.iter().enumerate().max_by_key
            }
        }
         */

        impl<T> $name<T> {
            pub const fn new(x: T, $($field: T,)+) -> Self {
                Self { x, $($field,)+ }
            }

            /// Tranpose the vector into a [ColumnVector]
            pub fn tranpose(self) -> ColumnVector<T, { count!(x, $( $field ),+) }> {
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

        impl<A, B> Mul<B> for $name<A>
        where
            A: Mul<B>,
            B: Clone,
        {
            type Output = $name<A::Output>;

            fn mul(self, rhs: B) -> $name<A::Output> {
                $name {
                    $( $field: self.$field * rhs.clone(), )+
                        x: self.x * rhs
                }
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
                $( self.$field * rhs.$field + )+ T::zero()
                // self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

implement_vector!(Vector2, x, y);

/*
impl<T> VectorSpace for Vector2<T> {
    type Scalar = T;
}

impl<T> MetricSpace for Vector2<T> {
    type Metric = T;

    fn distance2(self, other: Self) -> T {
        todo!()
    }
}

impl<T> InnerSpace for Vector2<T> {
    fn dot(self, other: Self) -> T {
        todo!()
    }
}
*/

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/*

impl<T> VectorSpace for Vector3<T> {
    type Scalar = T;
}

impl<T> MetricSpace for Vector3<T> {
    type Metric = T;

    fn distance2(self, other: Self) -> T {
        todo!()
    }
}

impl<T> InnerSpace for Vector3<T> {
    fn dot(self, other: Self) -> T {
        todo!()
    }
}

 */

impl<T> Vector3<T>
where
    T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Clone,
{
    pub fn cross(self, rhs: Vector3<T>) -> Self {
        let Vector3 { x: x0, y: y0, z: z0 } = self;
        let Vector3 { x: x1, y: y1, z: z1 } = rhs;
        Self {
            x: y0.clone() * z1.clone() - z0.clone() * y1.clone(),
            y: z0 * x1.clone() - x0.clone() * z1,
            z: x0 * y1 - y0 * x1,
        }
    }
}

implement_vector!(Vector3, x, y, z);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Vector4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Vector4<T> {
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

/*
impl<T> Vector4<T>
where
    T: Clone + PartialOrd,
{
    /// Return the largest value found in the vector, along with the associated index. If there is
    /// no largest value returns the first value ([Vector4::x]).
    pub fn argmax(&self) -> (usize, T) {
        let mut i_max = 0;
        let mut v_max = self.x.clone();
        if self.y > v_max {
            i_max = 1;
            v_max = self.y.clone();
        }
        if self.z > v_max {
            i_max = 2;
            v_max = self.z.clone();
        }
        if self.w > v_max {
            i_max = 3;
            v_max = self.w.clone();
        }
        (i_max, v_max)
    }

    /// Return the largest value in the vector. If there is no largest value, returns the first value
    /// ([Vector4::x]).
    pub fn max(&self) -> T {
        let mut v_max = self.x.clone();
        if self.y > v_max {
            v_max = self.y.clone();
        }
        if self.z > v_max {
            v_max = self.z.clone();
        }
        if self.w > v_max {
            v_max = self.w.clone();
        }
        v_max
    }

    /// Return the largest value found in the vector, along with the associated index. If there is
    /// no largest value returns the first value ([Vector4::x]).
    pub fn argmax(&self) -> (usize, T) {
        let mut i_max = 0;
        let mut v_max = self.x.clone();
        if self.y > v_max {
            i_max = 1;
            v_max = self.y.clone();
        }
        if self.z > v_max {
            i_max = 2;
            v_max = self.z.clone();
        }
        if self.w > v_max {
            i_max = 3;
            v_max = self.w.clone();
        }
        (i_max, v_max)
    }

    /// Return the largest value in the vector. If there is no largest value, returns the first value
    /// ([Vector4::x]).
    pub fn max(&self) -> T {
        let mut v_max = self.x.clone();
        if self.y > v_max {
            v_max = self.y.clone();
        }
        if self.z > v_max {
            v_max = self.z.clone();
        }
        if self.w > v_max {
            v_max = self.w.clone();
        }
        v_max
    }
}
*/

implement_vector!(Vector4, x, y, z, w);
