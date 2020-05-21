use aljabar::{matrix, vector, Matrix, Vector};

#[test]
fn test_serialize() {
    let v = vector![1u32, 2, 3, 4, 5, 6, 7];
    assert_eq!(serde_json::to_string(&v).unwrap(), "[1,2,3,4,5,6,7]");
    let m = matrix![[1u32, 2], [3u32, 4],];
    assert_eq!(serde_json::to_string(&m).unwrap(), "[[1,3],[2,4]]");
}

// Doesn't currently work due to a compiler bug.
#[test]
fn test_deserialize() {
    let v: Vector<u32, 7> = serde_json::from_str(&"[1,2,3,4,5,6,7]").unwrap();
    assert_eq!(v, vector![1u32, 2, 3, 4, 5, 6, 7],);
    let m: Matrix<u32, 2, 2> = serde_json::from_str(&"[[1,3],[2,4]]").unwrap();
    assert_eq!(m, matrix![[1u32, 2], [3u32, 4],],);
}
