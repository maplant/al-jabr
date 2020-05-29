//! Support for iterating over matrix rows. This is less natural than
//! iterating over columns due to the iteration not matching the stride
//! of the underlying storage.

use super::*;

/// A view into a given row of a matrix. It's possible to index just like a
/// normal Vector, but doesn't support the usual operators. It can only be
/// converted to a a Vector if the scalar value supports Clone.
pub struct RowView<'a, T, const N: usize, const M: usize> {
    row:    usize,
    matrix: &'a Matrix<T, { N }, { M }>,
}

impl<'a, T, const N: usize, const M: usize> RowView<'a, T, { N }, { M }> {
    /// Returns an iterator over the given row.
    pub fn iter(&self) -> impl Iterator<Item = &'a T> {
        RowViewIter {
            row:    self.row,
            col:    0,
            matrix: self.matrix,
        }
    }
}

impl<'a, T, const N: usize, const M: usize> IntoIterator for RowView<'a, T, { N }, { M }> {
    type Item = &'a T;
    type IntoIter = RowViewIter<'a, T, { N }, { M }>;

    fn into_iter(self) -> Self::IntoIter {
        RowViewIter {
            row:    self.row,
            col:    0,
            matrix: self.matrix,
        }
    }
}

impl<'a, T, const N: usize, const M: usize> Index<usize> for RowView<'a, T, { N }, { M }> {
    type Output = T;

    fn index(&self, column: usize) -> &Self::Output {
        &self.matrix[column][self.row]
    }
}

/// An iterator over a [RowView].
pub struct RowViewIter<'a, T, const N: usize, const M: usize> {
    row:    usize,
    col:    usize,
    matrix: &'a Matrix<T, { N }, { M }>,
}

impl<'a, T, const N: usize, const M: usize> Iterator for RowViewIter<'a, T, { N }, { M }> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let col = if self.col < M {
            self.col
        } else {
            return None;
        };
        self.col += 1;
        Some(&self.matrix[col][self.row])
    }
}

pub(super) struct RowIter<'a, T, const N: usize, const M: usize> {
    pub(super) row:    usize,
    pub(super) matrix: &'a Matrix<T, { N }, { M }>,
}

impl<'a, T, const N: usize, const M: usize> Iterator for RowIter<'a, T, { N }, { M }> {
    type Item = RowView<'a, T, { N }, { M }>;

    fn next(&mut self) -> Option<Self::Item> {
        let row = if self.row < N {
            Some(RowView {
                row:    self.row,
                matrix: self.matrix,
            })
        } else {
            None
        };
        self.row += 1;
        row
    }
}

/// A mutable view into a given row of a matrix. It's possible to index just
/// like a normal Vector, but doesn't support the usual operators. It can only
/// be converted to a Vector if the scalar value supports Clone.
pub struct RowViewMut<'a, T, const N: usize, const M: usize> {
    pub(super) row:    usize,
    pub(super) matrix: &'a mut Matrix<T, { N }, { M }>,
}

impl<'a, T, const N: usize, const M: usize> Index<usize> for RowViewMut<'a, T, { N }, { M }> {
    type Output = T;

    fn index(&self, column: usize) -> &Self::Output {
        &self.matrix[column][self.row]
    }
}

impl<'a, T, const N: usize, const M: usize> IndexMut<usize> for RowViewMut<'a, T, { N }, { M }> {
    fn index_mut(&mut self, column: usize) -> &mut Self::Output {
        &mut self.matrix[column][self.row]
    }
}

pub(super) struct RowIterMut<'a, T, const N: usize, const M: usize> {
    pub(super) row:     usize,
    pub(super) matrix:  *mut Matrix<T, { N }, { M }>,
    pub(super) phantom: PhantomData<&'a mut Matrix<T, { N }, { M }>>,
}

impl<'a, T, const N: usize, const M: usize> Iterator for RowIterMut<'a, T, { N }, { M }> {
    type Item = RowViewMut<'a, T, { N }, { M }>;

    fn next(&mut self) -> Option<Self::Item> {
        let row = if self.row < N {
            Some(RowViewMut {
                row:    self.row,
                matrix: unsafe { &mut *self.matrix },
            })
        } else {
            None
        };
        self.row += 1;
        row
    }
}
