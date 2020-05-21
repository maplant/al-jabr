extern crate aljabar;

use aljabar::{matrix, vector};

#[test]
fn can_use_vector_macro_outside_aljabar() {
    let _ = vector![1, 2, 3, 4, 5, 6];
}

#[test]
fn can_use_matrix_macro_outside_aljabar() {
    let _ = matrix![[1, 2, 3], [4, 5, 6],];
}
