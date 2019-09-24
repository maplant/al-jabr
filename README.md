# aljabar 

[![Documentation](https://docs.rs/aljabar/badge.svg)](https://docs.rs/aljabar)
[![Version](https://img.shields.io/crates/v/aljabar.svg)](https://crates.io/crates/aljabar)
[![Downloads](https://img.shields.io/crates/d/aljabar.svg)](https://crates.io/crates/aljabar)

An experimental n-dimensional linear algebra and mathematics library for the 
Rust programming language that makes extensive use of unstable Rust features.

aljabar supports Vectors and Matrices of any size and will provide 
implementations for any mathematic operations that are supported by their
scalars. Additionally, aljabar can leverage Rust's type system to ensure that
operations are only applied to values that are the correct size.

aljabar relies heavily on unstable Rust features such as const generics and thus
requires nightly to build. 

## Provided types

Currently, aljabar supports only `Vector` and `Matrix` types. Over the long 
term aljabar is intended to be feature compliant with [cgmath](https://github.com/rustgd/cgmath). 

### `Vector` 

Vectors can be constructed from arrays of any type and size. Use the `vector!`
macro to easily construct a vector:

```rust
let a = vector![ 0u32, 1, 2, 3 ]; 
assert_eq!(
    a, 
    Vector::<u32, 4>::from([ 0u32, 1, 2, 3 ])
);
```

`Add`, `Sub`, and `Neg` will be properly implemented for any `Vector<Scalar, N>`
for any respective implementation of such operations for `Scalar`. Operations 
are only implemented for vectors of equal sizes. 

```rust
let b = Vector::<f32, 7>::from([ 0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ]);
let c = Vector::<f32, 7>::from([ 1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]) * 0.5; 
assert_eq!(
    b + c, 
    Vector::<f32, 7>::from([ 0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 ])
);
```

If the scalar type implements `Mul` as well, then the Vector will be an 
`InnerSpace` and have the `dot` product defined for it, as well as the ability
to find the squared distance between two vectors (implements `MetricSpace`) and 
the squared magnitude of a vector. If the scalar type is a real number then the 
distance between two vectors and the magnitude of a vector can be found in 
addition:

```rust
let a = vector!(1i32, 1);
let b = vector!(5i32, 5);
assert_eq!(a.distance2(b), 32);       // distance method not implemented.
assert_eq!((b - a).magnitude2(), 32); // magnitude method not implemented.

let a = vector!(1.0f32, 1.0);
let b = vector!(5.0f32, 5.0);
const close: f32 = 5.65685424949;
assert_eq!(a.distance(b), close);       // distance is implemented.
assert_eq!((b - a).magnitude(), close); // magnitude is implemented.

// Vector normalization is also supported for floating point scalars.
assert_eq!(
    vector!(0.0f32, 20.0, 0.0)
        .normalize(),
    vector!(0.0f32, 1.0, 0.0)
);
```

#### Swizzling

[Swizzling](https://en.wikipedia.org/wiki/Swizzling_(computer_graphics))
is supported for both the `xyzw` or `rgba` conventions. All possible combinations
of the first four elements of a vector are supported. Single-element swizzle
functions return scalar results, multi-element swizzle functions return vector 
results of the appropriate size based on the number of selected elements.

```rust
let a = vector!(0.0f32, 1.0, 2.0, 3.0);

// A single element is returned as a scalar.
assert_eq!(1.0, a.y());

// Multiple elements are returned as a vector.
assert_eq!(vector!(2.0, 0.0, 3.0), a.zxw());

// The same element can be selected more than once.
assert_eq!(vector!(0.0f32, 0.0), a.rr());
```

Mixing `xyzw` and `rgba` swizzle conventions is not allowed.

```rust
let a = vector!(0.0f32, 1.0, 2.0, 3.0);
let b = a.rgzw(); // Does not compile.
```

Swizzling is supported on vectors of length less than 4. Attempting to access
elements past the length of the vector is a compile error.

```rust
let a = vector!(0.0f32, 1.0, 2.0);
let b = a.xyz(); // OK, only accesses the first 3 elements.
let c = a.rgba(); // Compile error, attempts to access missing 4th element.
```

### `Matrix`

Matrices can be created from arrays of vectors of any size and scalar type. 
Matrices are column-major and constructing a matrix from a raw array reflects 
that. The `matrix!` macro can be used to construct a matrix in row-major order:

```rust 
let a = Matrix::<f32, 3, 3>::from( [ vector!(1.0, 0.0, 0.0),
                                     vector!(0.0, 1.0, 0.0),
                                     vector!(0.0, 0.0, 1.0), ] );
let b: Matrix::<i32, 3, 3> = matrix![
    [ 0, -3, 5 ],
    [ 6, 1, -4 ],
    [ 2, 3, -2 ]
];
```

All operations performed on matrices produce fixed-size outputs. For example,
taking the `transpose` of a non-square matrix will produce a matrix with the 
width and height swapped: 

```rust
assert_eq!(
    Matrix::<i32, 1, 2>::from( [ vec1( 1 ), vec1( 2 ) ] )
        .transpose(),
    Matrix::<i32, 2, 1>::from( [ vec2( 1, 2 ) ] )
);
```

As with Vectors, if the underlying scalar type supports the appropriate 
operations, a `Matrix` will implement element-wise `Add` and `Sub` for matrices
of equal size: 

```rust 
let a = mat1x1( 1u32 );
let b = mat1x1( 2u32 );
let c = mat1x1( 3u32 );
assert_eq!(a + b, c);
```

And this is true for any type that implements `Add`, so therefore the following 
is possible as well:

```rust
let a = mat1x1(mat1x1(1u32));
let b = mat1x1(mat1x1(2u32));
let c = mat1x1(mat1x1(3u32));
assert_eq!(a + b, c);
```

For a given type `T`, if `T: Clone` and `Vector<T, _>` is an `InnerSpace`, then 
multiplication is defined for `Matrix<T, N, M> * Matrix<T, M, P>`. The result is
a `Matrix<T, N, P>`: 

```rust
let a: Matrix::<i32, 3, 3> = matrix![
    [ 0, -3, 5 ],
    [ 6, 1, -4 ],
    [ 2, 3, -2 ],
];
let b: Matrix::<i32, 3, 3> = matrix![
    [ -1, 0, -3 ],
    [  4, 5,  1 ],
    [  2, 6, -2 ],
];
let c: Matrix::<i32, 3, 3> = matrix![
    [  -2,  15, -13 ],
    [ -10, -19,  -9 ],
    [   6,   3,   1 ],
];
assert_eq!(
    a * b,
    c
);
```

It is important to note that this implementation is not necessarily commutative.
That is because matrices only need to share one dimension in order to be 
multiplied together in one direction. From this fact a somewhat pleasant trait 
bound appears: matrices that satisfy `Mul<Self>` *must* be square matrices. This
is reflected by a matrix's trait bounds; if a Matrix is square, you are able to 
extract a diagonal vector from it:

```rust
assert_eq!(
    matrix![
        [ 1i32, 0, 0 ],
        [    0, 2, 0 ],
        [    0, 0, 3 ],
    ].diagonal(),
    vector!(1i32, 2, 3) 
);

assert_eq!(
    matrix![
        [ 1i32, 0, 0, 0 ],
        [    0, 2, 0, 0 ],
        [    0, 0, 3, 0 ],
        [    0, 0, 0, 4 ],
    ].diagonal(),
    vector!(1i32, 2, 3, 4) 
);
```          


Matrices can be indexed by either their native column major storage or by
the more natural row major method. In order to use row-major indexing, call
`.index` or `.index_mut` on the matrix with a pair of indices. Calling 
`.index` with a single index will produce a Vector representing the
appropriate column of the matrix.

```rust
let m: Matrix::<i32, 2, 2> = matrix![
    [ 0, 2 ],
    [ 1, 3 ],
];

// Column-major indexing:
assert_eq!(m[0][0], 0);
assert_eq!(m[0][1], 1);
assert_eq!(m[1][0], 2);
assert_eq!(m[1][1], 3);

// Row-major indexing:
assert_eq!(m[(0, 0)], 0);
assert_eq!(m[(1, 0)], 1);
assert_eq!(m[(0, 1)], 2);
assert_eq!(m[(1, 1)], 3);
```

## Provided traits

### `Zero` and `One`

Defines the additive and multiplicative identity respectively. `zero` returns 
vectors and matrices filled with zeroes, while `one` returns the unit matrix 
and is unimplemented vectors.

### `Real`

As far as aljabar is considered, a `Real` is a value that has `sqrt` defined.
By default this is implemented for `f32` and `f64`.

### `VectorSpace`

If a `Vector` implements `Add` and `Sub` and its scalar implements `Mul` and 
`Div`, then that vector is part of a `VectorSpace`.

### `MetricSpace`

If two scalars have a distance squared, then vectors of that scalar type 
implement `MetricSpace`. 

### `RealMetricSpace`

If a vector is a `MetricSpace` and its scalar is a real number, then 
`.distance()` is defined as an alias for  `.distance2().sqrt()`

### `InnerSpace`

If a vector is a `VectorSpace`, a `MetricSpace`, and its scalar implements 
`Clone`, then it is also an `InnerSpace` and has the inner product defined for 
it. The inner product is known as the `dot` product and its respective method
in aljabar takes that spelling.

A vector that is an `InnerSpace` also has `.magnitude2()` implemented, which is 
defined as the length of the vector squared.

### `RealInnerSpace`

If a vector is an `InnerSpace` and its scalar is a real number then it inherits
a number of convenience methods: 

* `.magnitude()`: returns the length of the vector.
* `.normalize()`: returns a vector with the same direction and a length of one.
* `.normalize_to(magnitude)`: returns a vector with the same direction and a 
   length of `magnitude`.
* `.project_on(other)`: returns the vector projection of the current inner space
   onto the given argument.

### `SquareMatrix`

A `Matrix` that implements `Mul<Self>`. Square matrices have the following 
methods defined for them:

* `.diagonal()`: returns the diagonal vector of the matrix.
* `.determinant()`: returns the determinant of the matrix.
                    **currently only supported for matrices of `N <= 3`**
* `.invert()`: returns the inverse of the matrix if one exists. 
                    **currently only supported for matrices of `N <= 2`**

## Limitations

Individual component access isn't implemented for vectors and most likely will 
not be as simple as a public field access when it is. For now, manually 
destructure the vector or use `Index` and `IndexMut`:

```rust
let a = Vector1::<u32>::from([ 5 ]);
assert_eq!(a[0], 5);
let mut b = Vector2::<u32>::from([ 1, 2 ]);
b[1] += 3;
assert_eq!(b[1], 5);
```

Truncation must be performed by manually destructuring as well, but this is due
to a limitation of the current compiler.

## Contributions

Please feel free to submit pull requests of any nature.

## Future work 

There is a lot of work that needs to be done on aljabar before it can be 
considered stable or non-experimental.

* Implement `Matrix::invert`.
* Convert `Vector` ops to SIMD implementations.
* Add more tests and get code coverage to 100%.
* Add `Point` type.
* Implement faster matrix multiplication algorithms. 
