//! Support for rotational primitives.

use super::*;

// TODO: Bring this back for column vectors?
/*
/// A type that can rotate a [Vector] (or [Point]) of a given dimension.
pub trait Rotation<const DIMS: usize>
where
    Self: Sized,
{
    type Scalar;

    /// Returns the multiplicative inverse of the rotation. Effectively
    /// does the opposite of the given rotation.
    fn invert(&self) -> Self;

    /// Rotates a vector.
    fn rotate_vector(&self, v: ColumnVector<Self::Scalar, DIMS>) -> ColumnVector<Self::Scalar, DIMS>;

    /// Rotates a point around the origin.
    fn rotate_point(&self, p: Point<Self::Scalar, DIMS>) -> Point<Self::Scalar, DIMS> {
        let Matrix([res]) = self.rotate_vector(Matrix([p.0]));
        Point(res)
    }
}
 */

pub trait Rotation2 {
    type Scalar;

    /// Returns the multiplicative inverse of the rotation. Effectively
    /// does the opposite of the given rotation.
    fn invert(&self) -> Self;

    /// Rotates a vector.
    fn rotate_vector(&self, v: Vector2<Self::Scalar>) -> Vector2<Self::Scalar>;

    /// Rotates a point around the origin.
    fn rotate_point(&self, p: Point2<Self::Scalar>) -> Point2<Self::Scalar> {
        Point2::from_vec(self.rotate_vector(p.to_vec()))
    }
}

pub trait Rotation3 {
    type Scalar;

    /// Returns the multiplicative inverse of the rotation. Effectively
    /// does the opposite of the given rotation.
    fn invert(&self) -> Self;

    /// Rotates a vector.
    fn rotate_vector(&self, v: Vector3<Self::Scalar>) -> Vector3<Self::Scalar>;

    /// Rotates a point around the origin.
    fn rotate_point(&self, p: Point3<Self::Scalar>) -> Point3<Self::Scalar> {
        Point3::from_vec(self.rotate_vector(p.to_vec()))
    }
}

/// A representation of a rotation in three dimensional space. Each component is
/// the rotation around its respective axis in radians.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Euler<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// A [Matrix] that forms an orthonormal basis. Commonly known as a rotation
/// matrix.
pub struct Orthonormal<T, const DIMS: usize>(Matrix<T, DIMS, DIMS>);

impl<T, const DIMS: usize> Orthonormal<T, DIMS> {
    pub fn new(mat: Matrix<T, DIMS, DIMS>) -> Self {
        Self(mat)
    }
}

impl<T> From<T> for Orthonormal<T, 2>
where
    T: Real + Clone,
{
    fn from(angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        Orthonormal(Matrix([[c.clone(), s.clone()], [-s, c]]))
    }
}

impl<T> From<Euler<T>> for Orthonormal<T, 3>
where
    T: Real + Copy + Clone,
{
    fn from(Euler { x, y, z }: Euler<T>) -> Self {
        let ((xs, xc), (ys, yc), (zs, zc)) = (x.sin_cos(), y.sin_cos(), z.sin_cos());
        Orthonormal(Matrix([
            [yc * zc, xc * zs + xs * ys * zc, xs * zs - xc * ys * zc],
            [-yc * zs, xc * zc - xs * ys * zs, xs * zc + xc * ys * zs],
            [ys, -xs * yc, xc * yc],
        ]))
    }
}

impl<T, const DIMS: usize> Deref for Orthonormal<T, DIMS> {
    type Target = Matrix<T, DIMS, DIMS>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const DIMS: usize> DerefMut for Orthonormal<T, DIMS> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(feature = "serde")]
impl<T, const DIMS: usize> Serialize for Orthonormal<T, DIMS>
where
    Matrix<T, DIMS, DIMS>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T, const DIMS: usize> Deserialize<'de> for Orthonormal<T, DIMS>
where
    for<'a> Matrix<T, DIMS, DIMS>: Deserialize<'a>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Orthonormal(Matrix::<T, DIMS, DIMS>::deserialize(
            deserializer,
        )?))
    }
}

/*
impl<T, const DIMS: usize> Rotation<DIMS> for Orthonormal<T, DIMS>
where
    T: Clone + PartialOrd + Product + Real + One + Zero,
    T: Neg<Output = T>,
    T: Add<T, Output = T> + Sub<T, Output = T>,
    T: Mul<T, Output = T> + Div<T, Output = T>,
    Matrix<T, DIMS, DIMS>: Mul,
    Matrix<T, DIMS, DIMS>: Mul<ColumnVector<T, DIMS>, Output = ColumnVector<T, DIMS>>,
{
    type Scalar = T;

    fn invert(&self) -> Self {
        Orthonormal(self.0.clone().invert().unwrap())
    }

    fn rotate_vector(&self, v: ColumnVector<Self::Scalar, DIMS>) -> ColumnVector<Self::Scalar, DIMS> {
        self.0.clone() * v
    }
}
*/

/// A [quaternion](https://en.wikipedia.org/wiki/Quaternion), composed
/// of a scalar and a `Vector3`.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone)]
pub struct Quaternion<T> {
    pub s: T,
    pub v: Vector3<T>,
}

impl<T> Quaternion<T> {
    /// Constructs a Quaternion from one scalar and three imaginary
    /// components.
    pub const fn new(w: T, xi: T, yj: T, zk: T) -> Quaternion<T> {
        Quaternion {
            s: w,
            v: Vector3::new(xi, yj, zk),
        }
    }

    /// Constructs a Quaternion from a scalar and a vector.
    pub fn from_sv(s: T, v: Vector3<T>) -> Quaternion<T> {
        Quaternion { s, v }
    }
}

impl<T> Quaternion<T>
where
    T: Real + Clone,
{
    /// Constructs the rotation as a rotation along an axis.
    pub fn from_axis_angle(axis: Vector3<T>, angle: T) -> Self {
        let (s, c) = (angle.div2()).sin_cos();
        Quaternion::from_sv(c, axis * s)
    }
}

impl<T> From<Orthonormal<T, 3>> for Quaternion<T>
where
    T: Copy + Clone + PartialOrd + Product + Real + One + Zero,
    T: Neg<Output = T>,
    T: Add<T, Output = T> + Sub<T, Output = T>,
    T: Mul<T, Output = T> + Div<T, Output = T>,
{
    fn from(orthonormal: Orthonormal<T, 3>) -> Self {
        // Based on Glam, which is in turn based on https://github.com/microsoft/DirectXMath
        // `XM$quaternionRotationMatrix`
        let [m00, m01, m02] = orthonormal.0[0];
        let [m10, m11, m12] = orthonormal.0[1];
        let [m20, m21, m22] = orthonormal.0[2];
        if m22 <= T::zero() {
            // x^2 + y^2 >= z^2 + w^2
            let dif10 = m11 - m00;
            let omm22 = T::one() - m22;
            if dif10 <= T::zero() {
                // x^2 >= y^2
                let four_xsq = omm22 - dif10;
                let inv4x = four_xsq.sqrt().div2();
                Self::new(
                    four_xsq * inv4x,
                    (m01 + m10) * inv4x,
                    (m02 + m20) * inv4x,
                    (m12 - m21) * inv4x,
                )
            } else {
                // y^2 >= x^2
                let four_ysq = omm22 + dif10;
                let inv4y = four_ysq.sqrt().div2();
                Self::new(
                    (m01 + m10) * inv4y,
                    four_ysq * inv4y,
                    (m12 + m21) * inv4y,
                    (m20 - m02) * inv4y,
                )
            }
        } else {
            // z^2 + w^2 >= x^2 + y^2
            let sum10 = m11 + m00;
            let opm22 = T::one() + m22;
            if sum10 <= T::zero() {
                // z^2 >= w^2
                let four_zsq = opm22 - sum10;
                let inv4z = four_zsq.sqrt().div2();
                Self::new(
                    (m02 + m20) * inv4z,
                    (m12 + m21) * inv4z,
                    four_zsq * inv4z,
                    (m01 - m10) * inv4z,
                )
            } else {
                // w^2 >= z^2
                let four_wsq = opm22 + sum10;
                let inv4w = four_wsq.sqrt().div2();
                Self::new(
                    (m12 - m21) * inv4w,
                    (m20 - m02) * inv4w,
                    (m01 - m10) * inv4w,
                    four_wsq * inv4w,
                )
            }
        }
    }
}

impl<T> Quaternion<T>
where
    T: Zero + One,
{
    /// Return the identity quaternion.
    pub fn identity() -> Self {
        Quaternion::new(T::one(), T::zero(), T::zero(), T::zero())
    }
}

impl<T> Quaternion<T>
where
    T: Clone,
{
    pub fn s(&self) -> &T {
        &self.s
    }

    pub fn s_mut(&mut self) -> &mut T {
        &mut self.s
    }
}

impl<T> Quaternion<T>
where
    T: Neg<Output = T>,
{
    /// Returns the conjugate of the quaternion.
    pub fn conjugate(self) -> Self {
        Quaternion::from_sv(self.s, -self.v)
    }
}

impl<T> Zero for Quaternion<T>
where
    T: Zero,
{
    fn zero() -> Self {
        Quaternion::new(T::zero(), T::zero(), T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self.s.is_zero() && self.v.is_zero()
    }
}

impl<T> One for Quaternion<T>
where
    T: One,
{
    fn one() -> Self {
        Quaternion::new(T::one(), T::one(), T::one(), T::one())
    }

    fn is_one(&self) -> bool {
        self.s.is_one() && self.v.x.is_one() && self.v.y.is_one() && self.v.z.is_one()
    }
}

// Taken directly from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
impl From<Quaternion<f32>> for Euler<f32> {
    fn from(q: Quaternion<f32>) -> Euler<f32> {
        Euler {
            x: (2.0 * (q.s * q.v.x + q.v.y * q.v.z))
                .atan2(1.0 - 2.0 * (q.v.x * q.v.x + q.v.y * q.v.y)),
            y: {
                let sinp = 2.0 * (q.s * q.v.y - q.v.z * q.v.x);
                if sinp.abs() >= 1.0 {
                    std::f32::consts::FRAC_PI_2.copysign(sinp)
                } else {
                    sinp.asin()
                }
            },
            z: (2.0 * (q.s * q.v.z + q.v.x * q.v.y))
                .atan2(1.0 - 2.0 * (q.v.y * q.v.y + q.v.z * q.v.z)),
        }
    }
}

impl From<Quaternion<f64>> for Euler<f64> {
    fn from(q: Quaternion<f64>) -> Euler<f64> {
        Euler {
            x: (2.0 * (q.s * q.v.x + q.v.y * q.v.z))
                .atan2(1.0 - 2.0 * (q.v.x * q.v.x + q.v.y * q.v.y)),
            y: {
                let sinp = 2.0 * (q.s * q.v.y - q.v.z * q.v.x);
                if sinp.abs() >= 1.0 {
                    std::f64::consts::FRAC_PI_2.copysign(sinp)
                } else {
                    sinp.asin()
                }
            },
            z: (2.0 * (q.s * q.v.z + q.v.x * q.v.y))
                .atan2(1.0 - 2.0 * (q.v.y * q.v.y + q.v.z * q.v.z)),
        }
    }
}

impl<T> From<Euler<T>> for Quaternion<T>
where
    T: Real + Clone,
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

impl<T> Neg for Quaternion<T>
where
    T: Neg<Output = T>,
{
    type Output = Quaternion<T>;

    fn neg(self) -> Self {
        Self {
            s: -self.s,
            v: -self.v,
        }
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
            v: Vector3 { x, y, z },
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
            v: Vector3 { x, y, z },
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
        self.s = self.s().clone() * scalar.clone();
        self.v.x = self.v.x.clone() * scalar.clone();
        self.v.y = self.v.y.clone() * scalar.clone();
        self.v.z = self.v.z.clone() * scalar.clone();
    }
}

impl<T> DivAssign<T> for Quaternion<T>
where
    T: Real + Clone,
{
    fn div_assign(&mut self, scalar: T) {
        self.s = self.s().clone() / scalar.clone();
        self.v.x = self.v.x.clone() / scalar.clone();
        self.v.y = self.v.z.clone() / scalar.clone();
        self.v.z = self.v.y.clone() / scalar.clone();
    }
}

impl Mul<Quaternion<f32>> for f32 {
    type Output = Quaternion<f32>;

    fn mul(self, quat: Quaternion<f32>) -> Self::Output {
        quat * self
    }
}

impl Mul<Quaternion<f64>> for f64 {
    type Output = Quaternion<f64>;

    fn mul(self, quat: Quaternion<f64>) -> Self::Output {
        quat * self
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
            // Absolutely awful with all of the clones. I should probably
            // just require copy here.
            self.s().clone() * rhs.s().clone()
                - self.v.x.clone() * rhs.v.x.clone()
                - self.v.y.clone() * rhs.v.y.clone()
                - self.v.z.clone() * rhs.v.z.clone(),
            self.s().clone() * rhs.v.x.clone()
                + self.v.x.clone() * rhs.s.clone()
                + self.v.y.clone() * rhs.v.z.clone()
                - self.v.z.clone() * rhs.v.y.clone(),
            self.s().clone() * rhs.v.y.clone()
                + self.v.y.clone() * rhs.s.clone()
                + self.v.z.clone() * rhs.v.x.clone()
                - self.v.x.clone() * rhs.v.z.clone(),
            self.s().clone() * rhs.v.z.clone()
                + self.v.z.clone() * rhs.s.clone()
                + self.v.x.clone() * rhs.v.y.clone()
                - self.v.y.clone() * rhs.v.x.clone(),
        )
    }
}

impl<T> Mul<Vector3<T>> for Quaternion<T>
where
    T: Real + Clone,
{
    type Output = Vector3<T>;

    fn mul(self, rhs: Vector3<T>) -> Vector3<T> {
        let s = self.s().clone();
        self.v
            .clone()
            .cross(self.v.clone().cross(rhs.clone()) + (rhs.clone() * s))
            .map(Real::mul2)
            + rhs
    }
}

impl<T> Add<Quaternion<T>> for Quaternion<T>
where
    T: Add<T, Output = T>,
{
    type Output = Quaternion<T>;

    fn add(self, rhs: Quaternion<T>) -> Self {
        Quaternion::from_sv(self.s + rhs.s, self.v + rhs.v)
    }
}

impl<T> Sub<Quaternion<T>> for Quaternion<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Quaternion<T>;

    fn sub(self, rhs: Quaternion<T>) -> Self {
        Quaternion::from_sv(self.s - rhs.s, self.v - rhs.v)
    }
}

impl<T> VectorSpace for Quaternion<T>
where
    // These bounds can likely be reduced.
    T: Clone + Real + Zero,
{
    type Scalar = T;
}

impl<T> MetricSpace for Quaternion<T>
where
    T: Real + Zero + Clone,
{
    type Metric = T;

    fn distance2(self, other: Self) -> T {
        (other - self).magnitude2()
    }
}

impl<T> InnerSpace for Quaternion<T>
where
    T: Real + Zero + Clone,
{
    fn dot(self, other: Self) -> Self::Scalar {
        self.s * other.s + self.v.dot(other.v)
    }
}

impl<T> Rotation3 for Quaternion<T>
where
    T: Real + Clone + Zero + AddAssign,
{
    type Scalar = T;

    fn invert(&self) -> Self {
        self.clone().conjugate() / self.clone().magnitude2()
    }

    fn rotate_vector(&self, v: Vector3<T>) -> Vector3<T> {
        self.clone() * v
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
            self.s,
            self.v.x,
            self.v.y,
            self.v.z
        )
    }
}

impl<A, B> PartialEq<Quaternion<B>> for Quaternion<A>
where
    A: PartialEq<B>,
Vector3<A>: PartialEq<Vector3<B>>
{
    fn eq(&self, other: &Quaternion<B>) -> bool {
        self.s.eq(&other.s) && self.v.eq(&other.v)
    }
}

impl<T> Eq for Quaternion<T> where T: Eq {}

#[cfg(feature = "mint")]
impl<T: Copy> From<Quaternion<T>> for mint::Quaternion<T> {
    fn from(q: Quaternion<T>) -> mint::Quaternion<T> {
        mint::Quaternion {
            s: q.s,
            v: q.v.into(),
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Quaternion<T>> for Quaternion<T> {
    fn from(mint_quat: mint::Quaternion<T>) -> Self {
        Quaternion {
            s: mint_quat.s,
            v: Matrix([[mint_quat.v.x, mint_quat.v.y, mint_quat.v.z]]),
        }
    }
}
