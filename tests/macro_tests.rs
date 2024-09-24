extern crate al_jabr;

use al_jabr::{matrix, column_vector};

#[test]
fn can_use_vector_macro_outside_al_jabr() {
    let _ = column_vector![1, 2, 3, 4, 5, 6];
}

#[test]
fn can_use_matrix_macro_outside_al_jabr() {
    let _ = matrix![[1, 2, 3], [4, 5, 6],];
}
