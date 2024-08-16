use super::*;

/// An `N`-by-`M` Column Major matrix.
///
/// Matrices can be created from arrays of Vectors of any size and scalar type.
/// As with Vectors there are convenience constructor functions for square
/// matrices of the most common sizes.
///
/// ```
/// # use al_jabr::*;
/// let a = Matrix::<f32, 3, 3>::from( [ [ 1.0, 0.0, 0.0 ],
///                                      [ 0.0, 1.0, 0.0 ],
///                                      [ 0.0, 0.0, 1.0 ], ] );
/// let b: Matrix::<i32, 3, 3> = matrix![
///     [ 0, -3, 5 ],
///     [ 6, 1, -4 ],
///     [ 2, 3, -2 ]
/// ];
/// ```
///
/// All operations performed on matrices produce fixed-size outputs. For
/// example, taking the `transpose` of a non-square matrix will produce a matrix
/// with the width and height swapped:
///
/// ```
/// # use al_jabr::*;
/// assert_eq!(
///     Matrix::<i32, 1, 2>::from( [ [ 1 ], [ 2 ] ] )
///         .transpose(),
///     Matrix::<i32, 2, 1>::from( [ [ 1, 2 ] ] )
/// );
/// ```
///
/// # Indexing
///
/// Matrices can be indexed by either their native column major storage or by
/// the more natural row major method. In order to use row-major indexing, call
/// `.index` or `.index_mut` on the matrix with a pair of indices. Calling
/// `.index` with a single index will produce a vector representing the
/// appropriate column of the matrix.
///
/// ```
/// # use al_jabr::*;
/// let m: Matrix::<i32, 2, 2> = matrix![
///     [ 0, 2 ],
///     [ 1, 3 ],
/// ];
///
/// // Column-major indexing:
/// assert_eq!(m[0][0], 0);
/// assert_eq!(m[0][1], 1);
/// assert_eq!(m[1][0], 2);
/// assert_eq!(m[1][1], 3);
///
/// // Row-major indexing:
/// assert_eq!(m[(0, 0)], 0);
/// assert_eq!(m[(1, 0)], 1);
/// assert_eq!(m[(0, 1)], 2);
/// assert_eq!(m[(1, 1)], 3);
/// ```
///
/// # Iterating
///
/// Matrices are iterated most naturally over their columns, for which the
/// following three functions are provided:
///
/// * [column_iter](Matrix::column_iter), for immutably iterating over columns.
/// * [column_iter_mut](Matrix::column_iter_mut), for mutably iterating over
///   columns.
/// * [into_iter](IntoIterator::into_iter), for taking ownership of the columns.
///
/// Matrices can also be iterated over by their rows, however they can only
/// be iterated over by [RowViews](RowView), as they are not the natural
/// storage for Matrices. The following functions are provided:
///
/// * [row_iter](Matrix::row_iter), for immutably iterating over row views.
/// * [row_iter_mut](Matrix::row_iter_mut), for mutably iterating over row views
///   ([RowViewMut]).
/// * In order to take ownership of the rows of the matrix, `into_iter` should
///   called on the result of a [transpose](Matrix::transpose).
#[repr(transparent)]
pub struct Matrix<T, const N: usize, const M: usize>(pub(crate) [[T; N]; M]);

impl<T, const N: usize, const M: usize> Matrix<T, N, M> {
    /// Swap the two given columns in-place.
    pub fn swap_columns(&mut self, a: usize, b: usize) {
        unsafe { core::ptr::swap(&mut self.0[a], &mut self.0[b]) };
    }

    /// Swap the two given rows in-place.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        for v in self.0.iter_mut() {
            unsafe { core::ptr::swap(&mut v[a], &mut v[b]) };
        }
    }

    /// Swap the two given elements at index `a` and index `b`.
    ///
    /// The indices are expressed in the form `(column, row)`, which may be
    /// confusing given the indexing strategy for matrices.
    pub fn swap_elements(&mut self, (acol, arow): (usize, usize), (bcol, brow): (usize, usize)) {
        unsafe { core::ptr::swap(&mut self[acol][arow], &mut self[bcol][brow]) };
    }

    /// Returns an immutable iterator over the columns of the matrix.
    pub fn column_iter(&self) -> impl Iterator<Item = &'_ [T; N]> {
        self.0.iter()
    }

    /// Returns a mutable iterator over the columns of the matrix.
    pub fn column_iter_mut(&mut self) -> impl Iterator<Item = &'_ mut [T; N]> {
        self.0.iter_mut()
    }

    /// Returns an immutable iterator over the rows of the matrix.
    pub fn row_iter(&self) -> impl Iterator<Item = RowView<'_, T, N, M>> {
        RowIter {
            row: 0,
            matrix: self,
        }
    }

    /// Returns a mutable iterator over the rows of the matrix
    pub fn row_iter_mut(&mut self) -> impl Iterator<Item = RowViewMut<'_, T, N, M>> {
        RowIterMut {
            row: 0,
            matrix: self,
            phantom: PhantomData,
        }
    }

    /// Applies the given function to each element of the matrix, constructing a
    /// new matrix with the returned outputs.
    pub fn map<Out, F>(self, mut f: F) -> Matrix<Out, N, M>
    where
        F: FnMut(T) -> Out,
    {
        let mut from = MaybeUninit::new(self);
        let mut to = MaybeUninit::<Matrix<Out, N, M>>::uninit();
        let fromp: *mut MaybeUninit<[T; N]> =
            &mut from as *mut MaybeUninit<Matrix<T, N, M>> as *mut MaybeUninit<[T; N]>;
        let top: *mut [Out; N] = &mut to as *mut MaybeUninit<Matrix<Out, N, M>> as *mut [Out; N];
        for i in 0..M {
            unsafe {
                let fromp: *mut MaybeUninit<T> = mem::transmute(fromp.add(i));
                let top: *mut Out = mem::transmute(top.add(i));
                for j in 0..N {
                    top.add(j)
                        .write(f(fromp.add(j).replace(MaybeUninit::uninit()).assume_init()));
                }
            }
        }
        unsafe { to.assume_init() }
    }

    /// Returns the transpose of the matrix.
    pub fn transpose(self) -> Matrix<T, M, N> {
        let mut from = MaybeUninit::new(self);
        let mut trans = MaybeUninit::<[[T; M]; N]>::uninit();
        let fromp = &mut from as *mut MaybeUninit<Matrix<T, N, M>> as *mut [MaybeUninit<T>; N];
        let transp = &mut trans as *mut MaybeUninit<[[T; M]; N]> as *mut [T; M];
        for j in 0..N {
            // Fetch the current row
            let mut row = MaybeUninit::<[T; M]>::uninit();
            let rowp = &mut row as *mut MaybeUninit<[T; M]> as *mut T;
            for k in 0..M {
                unsafe {
                    let fromp: *mut MaybeUninit<T> = mem::transmute(fromp.add(k));
                    rowp.add(k)
                        .write(fromp.add(j).replace(MaybeUninit::uninit()).assume_init());
                }
            }
            let row = unsafe { row.assume_init() };
            unsafe {
                transp.add(j).write(row);
            }
        }
        Matrix::<T, M, N>(unsafe { trans.assume_init() })
    }
}

impl<T> Matrix3<T>
where
    T: Zero + One + Clone,
{
    /// Create an affine transformation matrix from a translation vector.
    pub fn from_translation(v: Vector2<T>) -> Self {
        let Matrix([[x, y]]) = v;
        Self([
            [T::one(), T::zero(), T::zero(),],
            [T::zero(), T::one(), T::zero(),],
            [x, y, T::one()],
        ])
    }

    /// Create an affine transformation matrix from a uniform scale.
    pub fn from_scale(scale: T) -> Self {
        Matrix3::from_nonuniform_scale(scale.clone(), scale)
    }

    /// Create an affine transformation matrix from a non-uniform scale.
    pub fn from_nonuniform_scale(x: T, y: T) -> Self {
        Self([
            [x, T::zero(), T::zero()],
            [T::zero(), y, T::zero()],
            [T::zero(), T::zero(), T::one()],
        ])
    }
}

impl<T> Matrix3<T>
where
    T: Real + One + Copy + Clone,
{
    /// Create an affine transformation matrix from a unit quaternion.
    pub fn from_rotation(q: Unit<Quaternion<T>>) -> Self {
        Self::from(q.into_inner())
    }
}

impl<T> Matrix4<T>
where
    T: Zero + One + Clone,
{
    /// Create an affine transformation matrix from a translation vector.
    pub fn from_translation(v: Vector3<T>) -> Self {
        let Matrix([[x, y, z]]) = v;
        Self([
            [T::one(), T::zero(), T::zero(), T::zero()],
            [T::zero(), T::one(), T::zero(), T::zero()],
            [T::zero(), T::zero(), T::one(), T::zero()],
            [x, y, z, T::one()],
        ])
    }

    /// Create an affine transformation matrix from a uniform scale.
    pub fn from_scale(scale: T) -> Self {
        Matrix4::from_nonuniform_scale(scale.clone(), scale.clone(), scale)
    }

    /// Create an affine trasnformation matrix from a non-uniform scale.
    pub fn from_nonuniform_scale(x: T, y: T, z: T) -> Self {
        Self([
            [x, T::zero(), T::zero(), T::zero()],
            [T::zero(), y, T::zero(), T::zero()],
            [T::zero(), T::zero(), z, T::zero()],
            [T::zero(), T::zero(), T::zero(), T::one()],
        ])
    }
}

impl<T> Matrix4<T>
where
    T: Real + One + Zero + Copy + Clone,
{
    /// Create an affine transformation matrix from a unit quaternion.
    pub fn from_rotation(q: Unit<Quaternion<T>>) -> Self {
        Self::from(q.into_inner())
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    T: Clone,
{
    /// Return the diagonal of the matrix. Only available for square matrices.
    pub fn diagonal(&self) -> Vector<T, N> {
        let mut diag = MaybeUninit::<[T; N]>::uninit();
        let diagp = &mut diag as *mut MaybeUninit<[T; N]> as *mut T;
        for i in 0..N {
            unsafe {
                diagp.add(i).write(self.0[i][i].clone());
            }
        }
        unsafe { Matrix([diag.assume_init()]) }
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    T: Clone + PartialOrd + Product + Real + One + Zero,
    T: Neg<Output = T>,
    T: Add<T, Output = T> + Sub<T, Output = T>,
    T: Mul<T, Output = T> + Div<T, Output = T>,
    Self: Add<Self>,
    Self: Sub<Self>,
    Self: Mul<Self>,
    Self: Mul<Vector<T, N>, Output = Vector<T, N>>,
{
    /// Returns the [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition) of
    /// the matrix, if one exists.
    pub fn lu(mut self) -> Option<LU<T, N>> {
        let mut p = Permutation::<N>::unit();

        for i in 0..N {
            let mut max_a = T::zero();
            let mut imax = i;
            for k in i..N {
                let abs = self.0[i][k].clone().abs();
                if abs > max_a {
                    max_a = abs;
                    imax = k;
                }
            }

            /* Check if matrix is degenerate */
            if max_a.is_zero() {
                return None;
            }

            /* Pivot rows */
            if imax != i {
                p.swap(i, imax);
                self.swap_rows(i, imax);
            }

            for j in i + 1..N {
                self[(j, i)] = self[(j, i)].clone() / self[(i, i)].clone();
                for k in i + 1..N {
                    self[(j, k)] =
                        self[(j, k)].clone() - self[(j, i)].clone() * self[(i, k)].clone();
                }
            }
        }
        Some(LU(p, self))
    }

    /// Returns the [determinant](https://en.wikipedia.org/wiki/Determinant) of
    /// the matrix.
    pub fn determinant(&self) -> T {
        self.clone().lu().map_or(T::zero(), |x| x.determinant())
    }

    /// Attempt to invert the matrix. For square matrices greater in size than
    /// three, [LU] decomposition is guaranteed to be used.
    pub fn invert(self) -> Option<Self> {
        self.lu().map(|x| x.invert())
    }
}

impl<T> Matrix4<T>
where
    T: Copy + Clone + PartialOrd + Product + Real + One + Zero,
    T: Neg<Output = T>,
    T: Add<T, Output = T> + Sub<T, Output = T>,
    T: Mul<T, Output = T> + Div<T, Output = T>,
    Self: Add<Self>,
    Self: Sub<Self>,
    Self: Mul<Self>,
    Self: Mul<Vector4<T>, Output = Vector4<T>>,
{
    /// Takes an affine matrix and returns the decomposition of it.
    pub fn to_scale_rotation_translation(&self) -> (Vector3<T>, Quaternion<T>, Vector3<T>) {
        let det = self.determinant();

        let scale = vector!(
            Vector::from(self[0]).magnitude() * det.signum(),
            Vector::from(self[1]).magnitude(),
            Vector::from(self[2]).magnitude()
        );

        let inv_scale = vector!(
            T::one() / *scale.x(),
            T::one() / *scale.y(),
            T::one() / *scale.z()
        );

        let Matrix([[xx, xy, xz, _]]) = Vector::from(self.0[0]) * *inv_scale.x();
        let Matrix([[yx, yy, yz, _]]) = Vector::from(self.0[1]) * *inv_scale.y();
        let Matrix([[zx, zy, zz, _]]) = Vector::from(self.0[2]) * *inv_scale.z();

        let rotation = Quaternion::from(Orthonormal::new(Matrix::from([
            vector!(xx, xy, xz),
            vector!(yx, yy, yz),
            vector!(zx, zy, zz),
        ])));

        let [x, y, z, _] = self[3];
        let translation = vector!(x, y, z);

        (scale, rotation, translation)
    }
}

impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<T, N, M> {
    fn from(array: [[T; N]; M]) -> Self {
        Matrix::<T, N, M>(array)
    }
}

impl<T, const N: usize, const M: usize> From<[Vector<T, N>; M]> for Matrix<T, N, M> {
    fn from(array: [Vector<T, N>; M]) -> Self {
        // I really hope that this copy gets optimized away because there's no avoiding
        // it.
        let mut from = MaybeUninit::new(array);
        let mut to = MaybeUninit::<Matrix<T, N, M>>::uninit();
        let fromp =
            &mut from as *mut MaybeUninit<[Matrix<T, N, 1>; M]> as *mut MaybeUninit<Vector<T, N>>;
        let top = &mut to as *mut MaybeUninit<Matrix<T, N, M>> as *mut [T; N];
        for i in 0..M {
            unsafe {
                let fromp: *mut MaybeUninit<T> = mem::transmute(fromp.add(i));
                let top: *mut T = mem::transmute(top.add(i));
                for j in 0..N {
                    top.add(j)
                        .write(fromp.add(j).replace(MaybeUninit::uninit()).assume_init());
                }
            }
        }
        unsafe { to.assume_init() }
    }
}

impl<T> From<Quaternion<T>> for Matrix3<T>
where
    // This is really annoying to implement with
    T: Real + One + Copy + Clone,
{
    fn from(quat: Quaternion<T>) -> Self {
        // Taken from cgmath
        let x2 = *quat.v.x() + *quat.v.x();
        let y2 = *quat.v.y() + *quat.v.y();
        let z2 = *quat.v.z() + *quat.v.z();

        let xx2 = x2 * *quat.v.x();
        let xy2 = x2 * *quat.v.y();
        let xz2 = x2 * *quat.v.z();

        let yy2 = y2 * *quat.v.y();
        let yz2 = y2 * *quat.v.z();
        let zz2 = z2 * *quat.v.z();

        let sy2 = y2 * quat.s;
        let sz2 = z2 * quat.s;
        let sx2 = x2 * quat.s;

        Self([
            [T::one() - yy2 - zz2, xy2 + sz2, xz2 - sy2],
            [xy2 - sz2, T::one() - xx2 - zz2, yz2 + sx2],
            [xz2 + sy2, yz2 - sx2, T::one() - xx2 - yy2],
        ])
    }
}

impl<T> From<Quaternion<T>> for Matrix4<T>
where
    // This is really annoying to implement with
    T: Add + Mul + Sub + Real + Zero + One + Copy + Clone,
{
    fn from(quat: Quaternion<T>) -> Self {
        // Taken from cgmath
        let x2 = *quat.v.x() + *quat.v.x();
        let y2 = *quat.v.y() + *quat.v.y();
        let z2 = *quat.v.z() + *quat.v.z();

        let xx2 = x2 * *quat.v.x();
        let xy2 = x2 * *quat.v.y();
        let xz2 = x2 * *quat.v.z();

        let yy2 = y2 * *quat.v.y();
        let yz2 = y2 * *quat.v.z();
        let zz2 = z2 * *quat.v.z();

        let sy2 = y2 * quat.s;
        let sz2 = z2 * quat.s;
        let sx2 = x2 * quat.s;

        Self([
            [T::one() - yy2 - zz2, xy2 + sz2, xz2 - sy2, T::zero()],
            [xy2 - sz2, T::one() - xx2 - zz2, yz2 + sx2, T::zero()],
            [xz2 + sy2, yz2 - sx2, T::one() - xx2 - yy2, T::zero()],
            [T::zero(), T::zero(), T::zero(), T::one()],
        ])
    }
}

/// A 2-by-2 square matrix.
pub type Matrix2<T> = Matrix<T, 2, 2>;

/// A 3-by-3 square matrix.
pub type Matrix3<T> = Matrix<T, 3, 3>;

/// A 4-by-4 square matrix.
pub type Matrix4<T> = Matrix<T, 4, 4>;

/// Constructs a new matrix from an array, using the more visually natural row
/// major order. Necessary to help the compiler. Prefer calling the macro
/// `matrix!`, which calls `new_matrix` internally.
#[inline]
#[doc(hidden)]
pub fn new_matrix<T: Clone, const N: usize, const M: usize>(rows: [[T; M]; N]) -> Matrix<T, N, M> {
    Matrix::<T, M, N>::from(rows).transpose()
}

/// Construct a [Matrix] of any size. The matrix is specified in row-major
/// order, but this function converts it to al_jabr's native column-major order.
///
/// ```
/// # use al_jabr::*;
/// // `matrix` allows you to create a matrix using natural writing order (row-major).
/// let m1: Matrix<u32, 4, 3> = matrix![
///     [0, 1, 2],
///     [3, 4, 5],
///     [6, 7, 8],
///     [9, 0, 1],
/// ];
///
/// // The equivalent code using the From implementation is below.
/// let m2: Matrix<u32, 4, 3> = Matrix::<u32, 4, 3>::from([
///     [0, 3, 6, 9],
///     [1, 4, 7, 0],
///     [2, 5, 8, 1],
/// ]);
///
/// assert_eq!(m1, m2);
/// ```
#[macro_export]
macro_rules! matrix {
    ( $($rows:expr),* $(,)? ) => {
        $crate::new_matrix([
            $($rows),*
        ])
    };
}

impl<T, const N: usize, const M: usize> Clone for Matrix<T, N, M>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Matrix::<T, N, M>(self.0.clone())
    }
}

impl<T, const N: usize, const M: usize> Copy for Matrix<T, N, M> where T: Copy {}

impl<T, const N: usize, const M: usize> Deref for Matrix<T, N, M> {
    type Target = [[T; N]; M];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize, const M: usize> DerefMut for Matrix<T, N, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const N: usize, const M: usize> Hash for Matrix<T, N, M>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in 0..M {
            self.0[i].hash(state);
        }
    }
}

impl<T, const N: usize, const M: usize> FromIterator<T> for Matrix<T, N, M> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let mut new = MaybeUninit::<[[T; N]; M]>::uninit();
        let newp = &mut new as *mut MaybeUninit<[[T; N]; M]> as *mut [T; N];

        for i in 0..M {
            let mut newv = MaybeUninit::<[T; N]>::uninit();
            let newvp = &mut newv as *mut MaybeUninit<[T; N]> as *mut T;
            for j in 0..N {
                if let Some(next) = iter.next() {
                    unsafe { newvp.add(j).write(next) };
                } else {
                    panic!(
                        "too few items in iterator to create Matrix<_, {}, {}>",
                        N, M
                    );
                }
            }
            unsafe {
                newp.add(i)
                    .write(mem::replace(&mut newv, MaybeUninit::uninit()).assume_init());
            }
        }

        if iter.next().is_some() {
            panic!(
                "too many items in iterator to create Matrix<_, {}, {}>",
                N, M
            );
        }

        Matrix::<T, N, M>(unsafe { new.assume_init() })
    }
}

impl<T, const N: usize, const M: usize> FromIterator<[T; N]> for Matrix<T, N, M> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = [T; N]>,
    {
        let mut iter = iter.into_iter();
        let mut new = MaybeUninit::<[[T; N]; M]>::uninit();
        let newp = &mut new as *mut MaybeUninit<[[T; N]; M]> as *mut [T; N];

        for i in 0..M {
            if let Some(v) = iter.next() {
                unsafe {
                    newp.add(i).write(v);
                }
            } else {
                panic!(
                    "too few items in iterator to create Matrix<_, {}, {}>",
                    N, M
                );
            }
        }
        Matrix::<T, N, M>(unsafe { new.assume_init() })
    }
}

impl<T, const N: usize, const M: usize> IntoIterator for Matrix<T, N, M> {
    type Item = [T; N];
    type IntoIter = ArrayIter<[T; N], M>;

    fn into_iter(self) -> Self::IntoIter {
        let Matrix(array) = self;
        ArrayIter {
            array: MaybeUninit::new(array),
            pos: 0,
        }
    }
}

impl<T, const N: usize, const M: usize> Zero for Matrix<T, N, M>
where
    T: Zero,
{
    fn zero() -> Self {
        let mut zero_mat = MaybeUninit::<[[T; N]; M]>::uninit();
        let matp = &mut zero_mat as *mut MaybeUninit<[[T; N]; M]> as *mut [T; N];

        for i in 0..M {
            let top = unsafe { matp.add(i) as *mut T };
            for j in 0..N {
                unsafe {
                    top.add(j).write(T::zero());
                }
            }
        }

        Matrix::<T, N, M>(unsafe { zero_mat.assume_init() })
    }

    fn is_zero(&self) -> bool {
        for i in 0..M {
            for j in 0..N {
                if !self.0[i][j].is_zero() {
                    return false;
                }
            }
        }
        true
    }
}

/// Constructs a unit matrix.
impl<T, const N: usize> One for Matrix<T, N, N>
where
    T: Zero + One + Clone,
    Self: PartialEq<Self>,
{
    fn one() -> Self {
        let mut unit_mat = MaybeUninit::<[[T; N]; N]>::uninit();
        let matp = &mut unit_mat as *mut MaybeUninit<[[T; N]; N]> as *mut [T; N];
        for i in 0..N {
            let mut unit_vec = MaybeUninit::<[T; N]>::uninit();
            let vecp = &mut unit_vec as *mut MaybeUninit<[T; N]> as *mut T;
            for j in 0..i {
                unsafe {
                    vecp.add(j).write(<T as Zero>::zero());
                }
            }
            unsafe {
                vecp.add(i).write(<T as One>::one());
            }
            for j in (i + 1)..N {
                unsafe {
                    vecp.add(j).write(<T as Zero>::zero());
                }
            }
            unsafe {
                matp.add(i).write(unit_vec.assume_init());
            }
        }
        Matrix::<T, N, N>(unsafe { unit_mat.assume_init() })
    }

    fn is_one(&self) -> bool {
        self == &<Self as One>::one()
    }
}

impl<T, const N: usize, const M: usize> Index<usize> for Matrix<T, N, M> {
    type Output = [T; N];

    fn index(&self, column: usize) -> &Self::Output {
        &self.0[column]
    }
}

impl<T, const N: usize, const M: usize> IndexMut<usize> for Matrix<T, N, M> {
    fn index_mut(&mut self, column: usize) -> &mut Self::Output {
        &mut self.0[column]
    }
}

impl<T, const N: usize, const M: usize> Index<(usize, usize)> for Matrix<T, N, M> {
    type Output = T;

    fn index(&self, (row, column): (usize, usize)) -> &Self::Output {
        &self.0[column][row]
    }
}

impl<T, const N: usize, const M: usize> IndexMut<(usize, usize)> for Matrix<T, N, M> {
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut Self::Output {
        &mut self.0[column][row]
    }
}

impl<A, B, RHS, const N: usize, const M: usize> PartialEq<RHS> for Matrix<A, N, M>
where
    RHS: Deref<Target = [[B; N]; M]>,
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

/// I'm not quite sure how to format the debug output for a matrix.
impl<T, const N: usize, const M: usize> fmt::Debug for Matrix<T, N, M>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix [ ")?;
        for j in 0..M {
            write!(f, "[ ")?;
            for i in 0..N {
                write!(f, "{:?} ", self.0[j][i])?;
            }
            write!(f, "] ")?;
        }
        write!(f, "]")
    }
}

/// Element-wise addition of two equal sized matrices.
impl<A, B, const N: usize, const M: usize> Add<Matrix<B, N, M>> for Matrix<A, N, M>
where
    A: Add<B>,
{
    type Output = Matrix<<A as Add<B>>::Output, N, M>;

    fn add(self, rhs: Matrix<B, N, M>) -> Self::Output {
        let mut mat = MaybeUninit::<[[<A as Add<B>>::Output; N]; M]>::uninit();
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let matp = &mut mat as *mut MaybeUninit<[[<A as Add<B>>::Output; N]; M]>
            as *mut [<A as Add<B>>::Output; N];
        let lhsp = &mut lhs as *mut MaybeUninit<Matrix<A, N, M>> as *mut MaybeUninit<[A; N]>;
        let rhsp = &mut rhs as *mut MaybeUninit<Matrix<B, N, M>> as *mut MaybeUninit<[B; N]>;
        for i in 0..M {
            let top = unsafe { matp.add(i) as *mut <A as Add<B>>::Output };
            let lhsp = unsafe { lhsp.add(i) as *mut MaybeUninit<A> };
            let rhsp = unsafe { rhsp.add(i) as *mut MaybeUninit<B> };
            for j in 0..N {
                unsafe {
                    top.add(j).write(
                        lhsp.add(j).replace(MaybeUninit::uninit()).assume_init()
                            + rhsp.add(j).replace(MaybeUninit::uninit()).assume_init(),
                    );
                }
            }
        }
        Matrix::<<A as Add<B>>::Output, N, M>(unsafe { mat.assume_init() })
    }
}

impl<A, B, const N: usize, const M: usize> AddAssign<Matrix<B, N, M>> for Matrix<A, N, M>
where
    A: AddAssign<B>,
{
    fn add_assign(&mut self, rhs: Matrix<B, N, M>) {
        let mut rhs = MaybeUninit::new(rhs);
        let rhsp = &mut rhs as *mut MaybeUninit<Matrix<B, N, M>> as *mut MaybeUninit<[B; N]>;
        for i in 0..M {
            let rhsp = unsafe { rhsp.add(i) as *mut MaybeUninit<B> };
            for j in 0..N {
                self.0[i][j] += unsafe { rhsp.add(j).replace(MaybeUninit::uninit()).assume_init() };
            }
        }
    }
}

/// Element-wise subtraction of two equal sized matrices.
impl<A, B, const N: usize, const M: usize> Sub<Matrix<B, N, M>> for Matrix<A, N, M>
where
    A: Sub<B>,
{
    type Output = Matrix<<A as Sub<B>>::Output, N, M>;

    fn sub(self, rhs: Matrix<B, N, M>) -> Self::Output {
        let mut mat = MaybeUninit::<[[<A as Sub<B>>::Output; N]; M]>::uninit();
        let mut lhs = MaybeUninit::new(self);
        let mut rhs = MaybeUninit::new(rhs);
        let matp = &mut mat as *mut MaybeUninit<[[<A as Sub<B>>::Output; N]; M]>
            as *mut [<A as Sub<B>>::Output; N];
        let lhsp = &mut lhs as *mut MaybeUninit<Matrix<A, N, M>> as *mut MaybeUninit<[A; N]>;
        let rhsp = &mut rhs as *mut MaybeUninit<Matrix<B, N, M>> as *mut MaybeUninit<[B; N]>;
        for i in 0..M {
            let top = unsafe { matp.add(i) as *mut <A as Sub<B>>::Output };
            let lhsp = unsafe { lhsp.add(i) as *mut MaybeUninit<A> };
            let rhsp = unsafe { rhsp.add(i) as *mut MaybeUninit<B> };
            for j in 0..N {
                unsafe {
                    top.add(j).write(
                        lhsp.add(j).replace(MaybeUninit::uninit()).assume_init()
                            - rhsp.add(j).replace(MaybeUninit::uninit()).assume_init(),
                    );
                }
            }
        }
        Matrix::<<A as Sub<B>>::Output, N, M>(unsafe { mat.assume_init() })
    }
}

impl<A, B, const N: usize, const M: usize> SubAssign<Matrix<B, N, M>> for Matrix<A, N, M>
where
    A: SubAssign<B>,
{
    fn sub_assign(&mut self, rhs: Matrix<B, N, M>) {
        let mut rhs = MaybeUninit::new(rhs);
        let rhsp = &mut rhs as *mut MaybeUninit<Matrix<B, N, M>> as *mut MaybeUninit<[B; N]>;
        for i in 0..M {
            let rhsp = unsafe { rhsp.add(i) as *mut MaybeUninit<B> };
            for j in 0..N {
                self.0[i][j] -= unsafe { rhsp.add(j).replace(MaybeUninit::uninit()).assume_init() };
            }
        }
    }
}

impl<T, const N: usize, const M: usize> Neg for Matrix<T, N, M>
where
    T: Neg,
{
    type Output = Matrix<<T as Neg>::Output, N, M>;

    fn neg(self) -> Self::Output {
        let mut from = MaybeUninit::new(self);
        let mut mat = MaybeUninit::<[[<T as Neg>::Output; N]; M]>::uninit();
        let fromp = &mut from as *mut MaybeUninit<Matrix<T, N, M>> as *mut MaybeUninit<[T; N]>;
        let matp = &mut mat as *mut MaybeUninit<[[<T as Neg>::Output; N]; M]>
            as *mut [<T as Neg>::Output; N];
        for i in 0..M {
            let top = unsafe { matp.add(i) as *mut <T as Neg>::Output };
            let fromp = unsafe { fromp.add(i) as *mut MaybeUninit<T> };
            for j in 0..N {
                unsafe {
                    top.add(j).write(
                        fromp
                            .add(j)
                            .replace(MaybeUninit::uninit())
                            .assume_init()
                            .neg(),
                    );
                }
            }
        }
        Matrix::<<T as Neg>::Output, N, M>(unsafe { mat.assume_init() })
    }
}

impl<T, const N: usize, const M: usize, const P: usize> Mul<Matrix<T, M, { P }>> for Matrix<T, N, M>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Clone + std::fmt::Debug,
Vector<T, M>: InnerSpace,
<Vector<T, M> as VectorSpace>::Scalar: std::fmt::Debug,
{
    type Output = Matrix<<Vector<T, M> as VectorSpace>::Scalar, N, { P }>;

    fn mul(self, rhs: Matrix<T, M, { P }>) -> Self::Output {
        // It might not seem that Rust's type system is helping me at all here,
        // but that's absolutely not true. I got the arrays iterations wrong on
        // the first try and Rust was nice enough to inform me of that fact.
        let mut mat = MaybeUninit::<[[<Vector<T, M> as VectorSpace>::Scalar; N]; P]>::uninit();
        let matp = &mut mat as *mut MaybeUninit<[[<Matrix<T, M, 1> as VectorSpace>::Scalar; N]; P]>
            as *mut [<Vector<T, M> as VectorSpace>::Scalar; N];
        for i in 0..P {
            let mut column = MaybeUninit::<[<Vector<T, M> as VectorSpace>::Scalar; N]>::uninit();
            let columnp = &mut column
                as *mut MaybeUninit<[<Matrix<T, M, 1> as VectorSpace>::Scalar; N]>
                as *mut <Vector<T, M> as VectorSpace>::Scalar;
            for j in 0..N {
                // Fetch the current row:
                let mut row = MaybeUninit::<[T; M]>::uninit();
                let rowp = &mut row as *mut MaybeUninit<[T; M]> as *mut T;
                for k in 0..M {
                    unsafe {
                        rowp.add(k).write(self.0[k][j].clone());
                    }
                }
                let row = unsafe { row.assume_init() };
                unsafe {
                    columnp
                        .add(j)
                        .write(Matrix([row]).dot(Matrix([rhs.0[i].clone()])));
                }
            }
            /*
            let column = Vector::<<Vector<T, M> as VectorSpace>::Scalar, N>(unsafe {
                column.assume_init()
            });
             */
            let column = unsafe { column.assume_init() };
            unsafe {
                matp.add(i).write(column);
            }
        }
        Matrix::<<Vector<T, M> as VectorSpace>::Scalar, N, { P }>(unsafe { mat.assume_init() })
    }
}

/*
impl<T, const N: usize, const M: usize> Mul<Vector<T, M>> for Matrix<T, N, M>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Clone,
    Vector<T, M>: InnerSpace,
{
    type Output = Vector<<Vector<T, M> as VectorSpace>::Scalar, N>;

    fn mul(self, rhs: Vector<T, M>) -> Self::Output {
        let mut column =
            MaybeUninit::<[<Vector<T, M> as VectorSpace>::Scalar; N]>::uninit();
        let columnp: *mut <Vector<T, M> as VectorSpace>::Scalar =
            unsafe { mem::transmute(&mut column) };
        for j in 0..N {
            // Fetch the current row:
            let mut row = MaybeUninit::<[T; M]>::uninit();
            let rowp: *mut T = unsafe { mem::transmute(&mut row) };
            for k in 0..M {
                unsafe {
                    rowp.add(k).write(self.0[k].0[j].clone());
                }
            }
            let row = Vector::<T, M>::from(unsafe { row.assume_init() });
            unsafe {
                columnp.add(j).write(row.dot(rhs.clone()));
            }
        }
        unsafe { column.assume_init() }
    }
}
 */

/// Scalar multiplication
impl<T, const N: usize, const M: usize> Mul<T> for Matrix<T, N, M>
where
    T: Mul<T, Output = T> + Clone,
{
    type Output = Matrix<T, N, M>;

    fn mul(self, scalar: T) -> Self::Output {
        let mut mat = MaybeUninit::<[[T; N]; M]>::uninit();
        let matp = &mut mat as *mut MaybeUninit<[[T; N]; M]> as *mut [T; N];
        for i in 0..M {
            let top = unsafe { matp.add(i) as *mut T };
            for j in 0..N {
                unsafe {
                    top.add(j).write(self.0[i][j].clone() * scalar.clone());
                }
            }
        }
        Matrix::<T, N, M>(unsafe { mat.assume_init() })
    }
}

impl<T, const N: usize, const M: usize> MulAssign<T> for Matrix<T, N, M>
where
    T: MulAssign<T> + Clone,
{
    fn mul_assign(&mut self, scalar: T) {
        for i in 0..M {
            for j in 0..N {
                self.0[i][j] *= scalar.clone();
            }
        }
    }
}

/// Scalar division
impl<T, const N: usize, const M: usize> Div<T> for Matrix<T, N, M>
where
    T: Div<T, Output = T> + Clone,
{
    type Output = Matrix<T, N, M>;

    fn div(self, scalar: T) -> Self::Output {
        let mut mat = MaybeUninit::<[[T; N]; M]>::uninit();
        let matp = &mut mat as *mut MaybeUninit<[[T; N]; M]> as *mut [T; N];
        for i in 0..M {
            let top = unsafe { matp.add(i) as *mut T };
            for j in 0..N {
                unsafe {
                    top.add(j).write(self.0[i][j].clone() / scalar.clone());
                }
            }
        }
        Matrix::<T, N, M>(unsafe { mat.assume_init() })
    }
}

impl<T, const N: usize, const M: usize> DivAssign<T> for Matrix<T, N, M>
where
    T: DivAssign<T> + Clone,
{
    fn div_assign(&mut self, scalar: T) {
        for i in 0..M {
            for j in 0..N {
                self.0[i][j] /= scalar.clone();
            }
        }
    }
}

impl<const N: usize, const M: usize> Mul<Matrix<f32, N, M>> for f32 {
    type Output = Matrix<f32, N, M>;

    fn mul(self, mat: Matrix<f32, N, M>) -> Self::Output {
        mat.map(|x| x * self)
    }
}

impl<const N: usize, const M: usize> Mul<Matrix<f64, N, M>> for f64 {
    type Output = Matrix<f64, N, M>;

    fn mul(self, mat: Matrix<f64, N, M>) -> Self::Output {
        mat.map(|x| x * self)
    }
}

/// Permutation matrix created for LU decomposition.
#[derive(Copy, Clone)]
pub struct Permutation<const N: usize> {
    arr: [usize; N],
    num_swaps: usize,
}

impl<const N: usize> fmt::Debug for Permutation<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[ ")?;
        for i in 0..N {
            write!(f, "{:?} ", self.arr[i])?;
        }
        write!(f, "] ")
    }
}

impl<RHS, const N: usize> PartialEq<RHS> for Permutation<N>
where
    RHS: Deref<Target = [usize; N]>,
{
    fn eq(&self, other: &RHS) -> bool {
        for (a, b) in self.arr.iter().zip(other.deref().iter()) {
            if !a.eq(b) {
                return false;
            }
        }
        true
    }
}

impl<const N: usize> Deref for Permutation<N> {
    type Target = [usize; N];

    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}

impl<const N: usize> DerefMut for Permutation<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.arr
    }
}

impl<const N: usize> Permutation<N> {
    /// Returns the unit permutation.
    pub fn unit() -> Permutation<N> {
        let mut arr: [MaybeUninit<usize>; N] = unsafe {
            // SAFETY: This is how uninit_array is implemented.
            MaybeUninit::<[MaybeUninit<usize>; N]>::uninit().assume_init()
        };
        let arr = unsafe {
            for (i, element) in arr.iter_mut().enumerate() {
                *element = MaybeUninit::new(i);
            }
            transmute_copy::<_, _>(&arr)
        };
        Permutation { arr, num_swaps: 0 }
    }

    /// Swaps two rows and increments the number of swaps.
    pub fn swap(&mut self, a: usize, b: usize) {
        self.num_swaps += 1;
        self.arr.swap(a, b);
    }

    /// Returns the number of swaps that have occurred.
    pub fn num_swaps(&self) -> usize {
        self.num_swaps
    }
}

impl<T, const N: usize> Mul<Vector<T, N>> for Permutation<N>
where
    // The clone bound can be removed from
    // here at some point with better
    // written code.
    T: Clone,
{
    type Output = Vector<T, N>;

    fn mul(self, rhs: Vector<T, N>) -> Self::Output {
        (0..N).map(|i| rhs[0][self[i]].clone()).collect()
    }
}

/// The result of LU factorizing a square matrix with partial-pivoting.
#[derive(Copy, Clone, Debug)]
pub struct LU<T, const N: usize>(Permutation<N>, Matrix<T, N, N>);

impl<T, const N: usize> Index<(usize, usize)> for LU<T, N> {
    type Output = T;

    fn index(&self, (row, column): (usize, usize)) -> &Self::Output {
        &self.1[(row, column)]
    }
}

impl<T, const N: usize> LU<T, N>
where
    T: Clone
        + PartialEq
        + One
        + Zero
        + Product
        + Neg<Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>,
{
    /// Returns the permutation sequence of the factorization.
    pub fn p(&self) -> &Permutation<N> {
        &self.0
    }

    /// Solves the linear equation `self * x = b` and returns `x`.
    pub fn solve(&self, b: Vector<T, N>) -> Vector<T, N> {
        let Matrix([mut x]) = self.0 * b;
        for i in 0..N {
            for k in 0..i {
                x[i] = x[i].clone() - self[(i, k)].clone() * x[k].clone();
            }
        }

        for i in (0..N).rev() {
            for k in i + 1..N {
                x[i] = x[i].clone() - self[(i, k)].clone() * x[k].clone();
            }

            // TODO(map): Consider making DivAssign a requirement so that we
            // don't have to clone here.
            x[i] = x[i].clone() / self[(i, i)].clone();
        }
        Matrix([x])
    }

    /// Returns the determinant of the matrix.
    pub fn determinant(&self) -> T {
        let Matrix([arr]) = self.1.diagonal();
        // TODO: Replace with .into_iter() in the next edition.
        let det: T = arr.iter().cloned().product();
        if self.0.num_swaps % 2 == 1 {
            -det
        } else {
            det
        }
    }

    /// Returns the inverse of the matrix, which is certain to exist.
    pub fn invert(self) -> Matrix<T, N, N> {
        Matrix::<T, N, N>::one()
            .into_iter()
            .map(|col| {
                let Matrix([x]) = self.solve(Matrix([col]));
                x
            })
            .collect()
    }
}

#[cfg(feature = "rand")]
impl<T, const N: usize, const M: usize> Distribution<Matrix<T, N, M>> for Standard
where
    Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Matrix<T, N, M> {
        let mut rand = MaybeUninit::<[[T; N]; M]>::uninit();
        let randp = &mut rand as *mut MaybeUninit<[[T; N]; M]> as *mut [T; N];

        for i in 0..M {
            let top = unsafe { randp.add(i) as *mut T };
            for j in 0..N {
                unsafe {
                    top.add(j).write(self.sample(rng));
                }
            }
        }

        Matrix::<T, N, M>(unsafe { rand.assume_init() })
    }
}

#[cfg(feature = "serde")]
mod column {
    use crate::*;

    pub(crate) struct ColumnRef<'a, T, const N: usize>(pub(crate) &'a [T; N]);

    impl<T, const N: usize> Serialize for ColumnRef<'_, T, N>
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

    #[repr(transparent)]
    pub(crate) struct Column<T, const N: usize>([T; N]);

    impl<'de, T, const N: usize> Deserialize<'de> for Column<T, N>
    where
        T: Deserialize<'de>,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserializer
                .deserialize_tuple(N, ArrayVisitor::<[T; N]>::new())
                .map(Column)
        }
    }
}

#[cfg(feature = "serde")]
impl<T, const N: usize, const M: usize> Serialize for Matrix<T, N, M>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_tuple(M)?;
        for i in 0..M {
            seq.serialize_element(&column::ColumnRef(&self.0[i]))?;
        }
        seq.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, const N: usize, const M: usize> Deserialize<'de> for Matrix<T, N, M>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // This is a mess, but it is the only way I could think of to safely do this.
        let mut des = MaybeUninit::new(
            deserializer.deserialize_tuple(N, ArrayVisitor::<[column::Column<T, N>; M]>::new())?,
        );
        let inp = &mut des as *mut MaybeUninit<[column::Column<T, N>; M]>
            as *mut MaybeUninit<column::Column<T, N>>;
        let mut out = MaybeUninit::<[[T; N]; M]>::uninit();
        let outp = &mut out as *mut MaybeUninit<[[T; N]; M]> as *mut MaybeUninit<[T; N]>;
        for i in 0..M {
            let inp = unsafe { inp.add(i) as *mut MaybeUninit<T> };
            let outp = unsafe { outp.add(i) as *mut T };
            for j in 0..N {
                unsafe {
                    outp.add(j)
                        .write(inp.add(j).replace(MaybeUninit::uninit()).assume_init());
                }
            }
        }
        Ok(Matrix(unsafe { out.assume_init() }))
    }
}

#[cfg(any(feature = "approx", test))]
use approx::AbsDiffEq;

#[cfg(any(feature = "approx", test))]
impl<T: AbsDiffEq, const N: usize, const M: usize> AbsDiffEq for Matrix<T, N, M>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        for i in 0..M {
            for j in 0..N {
                if !T::abs_diff_eq(&self[i][j], &other[i][j], epsilon) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(feature = "approx")]
use approx::{RelativeEq, UlpsEq};

#[cfg(feature = "approx")]
impl<T, const N: usize, const M: usize> RelativeEq for Matrix<T, N, M>
where
    T: RelativeEq,
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        for i in 0..M {
            for j in 0..N {
                if !T::relative_eq(&self[i][j], &other[i][j], epsilon, max_relative) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(feature = "approx")]
impl<T, const N: usize, const M: usize> UlpsEq for Matrix<T, N, M>
where
    T: UlpsEq,
    T::Epsilon: Copy,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        for i in 0..M {
            for j in 0..N {
                if !T::ulps_eq(&self[i][j], &other[i][j], epsilon, max_ulps) {
                    return false;
                }
            }
        }
        true
    }
}

macro_rules! into_mint_column_matrix {
    ($mint_name:ident, $rows:expr, $cols:expr $( , ($col_name:ident, $col_idx:expr ) )+) => {
        #[cfg(feature = "mint")]
        impl<T: Copy> From<Matrix<T, {$rows}, {$cols}>> for mint::$mint_name<T> {
            fn from(m: Matrix<T, {$rows}, {$cols}>) -> Self {
                mint::$mint_name {
                    $(
                        $col_name: m.0[$col_idx].into(),
                    )*
                }
            }
        }
    }
}

into_mint_column_matrix!(ColumnMatrix2, 2, 2, (x, 0), (y, 1));
into_mint_column_matrix!(ColumnMatrix3, 3, 3, (x, 0), (y, 1), (z, 2));
into_mint_column_matrix!(ColumnMatrix4, 4, 4, (x, 0), (y, 1), (z, 2), (w, 3));
into_mint_column_matrix!(ColumnMatrix2x3, 2, 3, (x, 0), (y, 1), (z, 2));
into_mint_column_matrix!(ColumnMatrix2x4, 2, 4, (x, 0), (y, 1), (z, 2), (w, 3));
into_mint_column_matrix!(ColumnMatrix3x2, 3, 2, (x, 0), (y, 1));
into_mint_column_matrix!(ColumnMatrix3x4, 3, 4, (x, 0), (y, 1), (z, 2), (w, 3));
into_mint_column_matrix!(ColumnMatrix4x2, 4, 2, (x, 0), (y, 1));
into_mint_column_matrix!(ColumnMatrix4x3, 4, 3, (x, 0), (y, 1), (z, 2));

macro_rules! from_mint_column_matrix {
    ($mint_name:ident, $rows:expr, $cols:expr, $($component:ident),+) => {
        #[cfg(feature = "mint")]
        impl<T> From<mint::$mint_name<T>> for Matrix<T, {$rows}, {$cols}> {
            fn from(m: mint::$mint_name<T>) -> Self {
                Self([
                    $(
                        m.$component.into(),
                    )*
                ])
            }
        }
    }
}

from_mint_column_matrix!(ColumnMatrix2, 2, 2, x, y);
from_mint_column_matrix!(ColumnMatrix3, 3, 3, x, y, z);
from_mint_column_matrix!(ColumnMatrix4, 4, 4, x, y, z, w);
from_mint_column_matrix!(ColumnMatrix2x3, 2, 3, x, y, z);
from_mint_column_matrix!(ColumnMatrix2x4, 2, 4, x, y, z, w);
from_mint_column_matrix!(ColumnMatrix3x2, 3, 2, x, y);
from_mint_column_matrix!(ColumnMatrix3x4, 3, 4, x, y, z, w);
from_mint_column_matrix!(ColumnMatrix4x2, 4, 2, x, y);
from_mint_column_matrix!(ColumnMatrix4x3, 4, 3, x, y, z);

macro_rules! into_mint_row_matrix {
    ($mint_name:ident, $rows:expr, $cols:expr $( , ($col_name:ident, $col_idx:expr ) )+) => {
        #[cfg(feature = "mint")]
        impl<T: Copy> From<Matrix<T, {$rows}, {$cols}>> for mint::$mint_name<T>  {
            fn from(m: Matrix<T, {$rows}, {$cols}>) -> mint::$mint_name<T> {
                let transposed = m.transpose();
                mint::$mint_name {
                    $(
                        $col_name: transposed.0[$col_idx].into(),
                    )*
                }
            }
        }
    }
}

into_mint_row_matrix!(RowMatrix2, 2, 2, (x, 0), (y, 1));
into_mint_row_matrix!(RowMatrix3, 3, 3, (x, 0), (y, 1), (z, 2));
into_mint_row_matrix!(RowMatrix4, 4, 4, (x, 0), (y, 1), (z, 2), (w, 3));
into_mint_row_matrix!(RowMatrix2x3, 2, 3, (x, 0), (y, 1));
into_mint_row_matrix!(RowMatrix2x4, 2, 4, (x, 0), (y, 1));
into_mint_row_matrix!(RowMatrix3x2, 3, 2, (x, 0), (y, 1), (z, 2));
into_mint_row_matrix!(RowMatrix3x4, 3, 4, (x, 0), (y, 1), (z, 2));
into_mint_row_matrix!(RowMatrix4x2, 4, 2, (x, 0), (y, 1), (z, 2), (w, 3));
into_mint_row_matrix!(RowMatrix4x3, 4, 3, (x, 0), (y, 1), (z, 2), (w, 3));

// It would be possible to implement this without a runtime transpose() by
// directly copying the corresponding elements from the mint matrix to the
// appropriate position in the al_jabr matrix, but it would be substantially
// more code to do so. I'm leaving it as a transpose for now in the expectation
// that converting between al_jabr and mint entities will occur infrequently at
// program boundaries.
macro_rules! from_mint_row_matrix {
    ($mint_name:ident, $rows:expr, $cols:expr, $($component:ident),+) => {
        #[cfg(feature = "mint")]
        impl<T> From<mint::$mint_name<T>> for Matrix<T, {$rows}, {$cols}> {
            fn from(m: mint::$mint_name<T>) -> Self {
                Matrix::<T, {$cols}, {$rows}>([
                    $(
                        m.$component.into(),
                    )*
                ]).transpose()
            }
        }
    }
}

from_mint_row_matrix!(RowMatrix2, 2, 2, x, y);
from_mint_row_matrix!(RowMatrix3, 3, 3, x, y, z);
from_mint_row_matrix!(RowMatrix4, 4, 4, x, y, z, w);
from_mint_row_matrix!(RowMatrix2x3, 2, 3, x, y);
from_mint_row_matrix!(RowMatrix2x4, 2, 4, x, y);
from_mint_row_matrix!(RowMatrix3x2, 3, 2, x, y, z);
from_mint_row_matrix!(RowMatrix3x4, 3, 4, x, y, z);
from_mint_row_matrix!(RowMatrix4x2, 4, 2, x, y, z, w);
from_mint_row_matrix!(RowMatrix4x3, 4, 3, x, y, z, w);
