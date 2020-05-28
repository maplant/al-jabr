//! Support for rotational primitives.

use super::*;

/// A type that can rotate a `Vector` of a given dimension.
pub trait Rotation<const DIMS: usize>
where
    Self: Sized,
{
    type Scalar;

    fn rotate_vector(self, v: Vector<Self::Scalar, { DIMS }>) -> Vector<Self::Scalar, { DIMS }>;

    fn rotate_point(self, p: Point<Self::Scalar, { DIMS }>) -> Point<Self::Scalar, { DIMS }> {
        Point(self.rotate_vector(Vector(p.0)).0)
    }
}

/// A value for which the usual set of trigonometric functions are defined.
pub trait Angle: Real {
    /// Returns the sine of the angle.
    fn sin(self) -> Self;

    /// Returns the cosine of the angle.
    fn cos(self) -> Self;

    /// Returns the tangent of the angle.
    fn tan(self) -> Self;

    /// Returns the four quadrant arctangent of `self` and `x` in radians.
    fn atan2(self, x: Self) -> Self;

    /// Returns the sine and the cosine of the angle.
    fn sin_cos(self) -> (Self, Self);
}

impl Angle for f32 {
    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn atan2(self, x: Self) -> Self {
        self.atan2(x)
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}

impl Angle for f64 {
    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn atan2(self, x: Self) -> Self {
        self.atan2(x)
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}

/// A representation of a rotation in three dimensional space. Each component is
/// the rotation around its respective axis in radians.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Euler<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// A `Matrix` that forms an orthonormal basis. Commonly known as a rotation
/// matrix.
pub struct Orthonormal<T, const DIMS: usize>(Matrix<T, { DIMS }, { DIMS }>);

impl<T> From<T> for Orthonormal<T, 2>
where
    T: Angle + Clone,
{
    fn from(angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        Orthonormal(Matrix([vector!(c.clone(), s.clone()), vector!(-s, c)]))
    }
}

impl<T> From<Euler<T>> for Orthonormal<T, 3>
where
    T: Angle + Copy + Clone,
{
    fn from(Euler { x, y, z }: Euler<T>) -> Self {
        let ((xs, xc), (ys, yc), (zs, zc)) = (x.sin_cos(), y.sin_cos(), z.sin_cos());
        Orthonormal(Matrix([
            vector![yc * zc, xc * zs + xs * ys * zc, xs * zs - xc * ys * zc],
            vector![-yc * zs, xc * zc - xs * ys * zs, xs * zc + xc * ys * zs],
            vector![ys, -xs * yc, xc * yc],
        ]))
    }
}

impl<T, const DIMS: usize> Deref for Orthonormal<T, { DIMS }> {
    type Target = Matrix<T, { DIMS }, { DIMS }>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const DIMS: usize> DerefMut for Orthonormal<T, { DIMS }> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(feature = "serde")]
impl<T, const DIMS: usize> Serialize for Orthonormal<T, { DIMS }>
where
    Matrix<T, { DIMS }, { DIMS }>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T, const DIMS: usize> Deserialize<'de> for Orthonormal<T, { DIMS }>
where
    for<'a> Matrix<T, { DIMS }, { DIMS }>: Deserialize<'a>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Orthonormal(Matrix::<T, { DIMS }, { DIMS }>::deserialize(
            deserializer,
        )?))
    }
}

impl<T, const DIMS: usize> Rotation<{ DIMS }> for Orthonormal<T, { DIMS }>
where
    Matrix<T, { DIMS }, { DIMS }>: Mul<Vector<T, { DIMS }>, Output = Vector<T, { DIMS }>>,
{
    type Scalar = T;

    fn rotate_vector(self, v: Vector<Self::Scalar, { DIMS }>) -> Vector<Self::Scalar, { DIMS }> {
        self.0 * v
    }
}

/// A `Quaternion`, composed of a scalar and a `Vector3`.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone)]
pub struct Quaternion<T> {
    pub s: T,
    pub v: Vector3<T>,
}

impl<T> From<Euler<T>> for Quaternion<T>
where
    T: Angle + Clone,
{
    fn from(euler: Euler<T>) -> Quaternion<T> {
        let Euler { x, y, z } = euler;
        let (xs, xc) = x.div2().sin_cos();
        let (ys, yc) = y.div2().sin_cos();
        let (zs, zc) = z.div2().sin_cos();

        Quaternion::new(
            -xs.clone() * ys.clone() * zs.clone() + xc.clone() * yc.clone() * zc.clone(),
            xs.clone() * yc.clone() * zc.clone() + ys.clone() * zs.clone() * xc.clone(),
            -xs.clone() * zs.clone() * yc.clone() + ys.clone() * xc.clone() * zc.clone(),
            xs.clone() * ys.clone() * zc.clone() + zs.clone() * xc.clone() * yc.clone(),
        )
    }
}

impl<T> Quaternion<T> {
    pub const fn new(w: T, xi: T, yj: T, zk: T) -> Quaternion<T> {
        Quaternion {
            s: w,
            v: Vector([xi, yj, zk]),
        }
    }
}

impl<T> Quaternion<T>
where
    T: Clone,
{
    /// Alias for `.s.clone()`
    pub fn s(&self) -> T {
        self.s.clone()
    }
}

impl<T> Mul<T> for Quaternion<T>
where
    T: Real + Clone,
{
    type Output = Quaternion<T>;

    fn mul(self, scalar: T) -> Self {
        let Quaternion {
            s,
            v: Vector([x, y, z]),
        } = self;
        Quaternion::new(
            s * scalar.clone(),
            x * scalar.clone(),
            y * scalar.clone(),
            z * scalar,
        )
    }
}

impl<T> Div<T> for Quaternion<T>
where
    T: Real + Clone,
{
    type Output = Quaternion<T>;

    fn div(self, scalar: T) -> Self {
        let Quaternion {
            s,
            v: Vector([x, y, z]),
        } = self;
        Quaternion::new(
            s / scalar.clone(),
            x / scalar.clone(),
            y / scalar.clone(),
            z / scalar,
        )
    }
}

impl<T> MulAssign<T> for Quaternion<T>
where
    T: Real + Clone,
{
    fn mul_assign(&mut self, scalar: T) {
        self.s = self.s() * scalar.clone();
        self.v[0] = self.v[0].clone() * scalar.clone();
        self.v[1] = self.v[1].clone() * scalar.clone();
        self.v[2] = self.v[2].clone() * scalar.clone();
    }
}

impl<T> DivAssign<T> for Quaternion<T>
where
    T: Real + Clone,
{
    fn div_assign(&mut self, scalar: T) {
        self.s = self.s() / scalar.clone();
        self.v[0] = self.v[0].clone() / scalar.clone();
        self.v[1] = self.v[1].clone() / scalar.clone();
        self.v[2] = self.v[2].clone() / scalar.clone();
    }
}

impl<T> Mul<Quaternion<T>> for Quaternion<T>
where
    T: Real + Clone,
{
    type Output = Quaternion<T>;

    fn mul(self, rhs: Quaternion<T>) -> Self {
        Quaternion::new(
            // source: cgmath/quaternion.rs
            self.s() * rhs.s()
                - self.v.x() * rhs.v.x()
                - self.v.y() * rhs.v.y()
                - self.v.z() * rhs.v.z(),
            self.s() * rhs.v.x() + self.v.x() * rhs.s() + self.v.y() * rhs.v.z()
                - self.v.z() * rhs.v.y(),
            self.s() * rhs.v.y() + self.v.y() * rhs.s() + self.v.z() * rhs.v.x()
                - self.v.x() * rhs.v.z(),
            self.s() * rhs.v.z() + self.v.z() * rhs.s() + self.v.x() * rhs.v.y()
                - self.v.y() * rhs.v.x(),
        )
    }
}

impl<T> Mul<Vector3<T>> for Quaternion<T>
where
    T: Real + Clone,
{
    type Output = Vector3<T>;

    fn mul(self, rhs: Vector3<T>) -> Vector3<T> {
        let s = self.s();
        self.v
            .clone()
            .cross(self.v.clone().cross(rhs.clone()) + (rhs.clone() * s))
            .map(Real::mul2)
            + rhs
    }
}

impl<T> Rotation<3> for Quaternion<T>
where
    T: Real + Clone,
{
    type Scalar = T;

    fn rotate_vector(self, v: Vector<Self::Scalar, 3>) -> Vector<Self::Scalar, 3> {
        self * v
    }
}

impl<T> fmt::Debug for Quaternion<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Quaternion {{ s: {:?}, x: {:?}, y: {:?}, z: {:?} }}",
            self.s, self.v.0[0], self.v.0[1], self.v.0[2]
        )
    }
}

impl<A, B> PartialEq<Quaternion<B>> for Quaternion<A>
where
    A: PartialEq<B>,
{
    fn eq(&self, other: &Quaternion<B>) -> bool {
        self.s.eq(&other.s) && self.v.eq(&other.v)
    }
}

impl<T> Eq for Quaternion<T> where T: Eq {}

#[cfg(feature = "mint")]
impl<T: Copy> Into<mint::Quaternion<T>> for Quaternion<T> {
    fn into(self) -> mint::Quaternion<T> {
        mint::Quaternion {
            s: self.s,
            v: self.v.into(),
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Quaternion<T>> for Quaternion<T> {
    fn from(mint_quat: mint::Quaternion<T>) -> Self {
        Quaternion {
            s: mint_quat.s,
            v: Vector([mint_quat.v.x, mint_quat.v.y, mint_quat.v.z]),
        }
    }
}
