# aljabar 

An experimental n-dimensional linear algebra and mathematics library for the 
Rust programming language that makes extensive use of unstable Rust features.

aljabar supports Vectors and Matrices of any size and will provide 
implementations for any mathematic operations that are supported by their
scalars. Additionally, aljabar can leverage Rust's type system to ensure that
operations are only applied to values that are the correct size.

aljabar is very incomplete and not entirely safe in its current form. If your 
scalar has the possibility of panicking during a math operation, aljabar is not 
guaranteed to clean up properly. Division by zero should be avoided.

aljabar relies heavily on unstable Rust features such as const generics and thus
requires nightly to build. 

## Provided types

Currently, aljabar supports only `Vector` and `Matrix` types. Over the long 
term aljabar is intended to be feature compliant with [cgmath](https://github.com/rustgd/cgmath). 

### `Vector` 

Vectors can be constructed from arrays of any type and size. There are 
convenience constructor functions provided for the most common sizes:

```rust
let a = vec4( 0u32, 1, 2, 3 ); 
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
// Construct a unit vector and multiply each element by 1/2:
let c = Vector::<f32, 7>::one() * 0.5; 
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
let a = vec2( 1i32, 1);
let b = vec2( 5i32, 5 );
assert_eq!(a.distance2(b), 32);       // distance method not implemented.
assert_eq!((b - a).magnitude2(), 32); // magnitude method not implemented.

let a = vec2( 1.0f32, 1.0 );
let b = vec2( 5.0f32, 5.0 );
const close: f32 = 5.65685424949;
assert_eq!(a.distance(b), close);       // distance is implemented.
assert_eq!((b - a).magnitude(), close); // magnitude is implemented.

// Vector normalization is also supported for floating point scalars.
assert_eq!(
    vec3( 0.0f32, 20.0, 0.0 )
        .normalize(),
    vec3( 0.0f32, 1.0, 0.0 )
);
```

### `Matrix`

Matrices can be created from arrays of Vectors of any size and scalar type. 
As with Vectors there are convenience constructor functions for square matrices
of the most common sizes: 

```rust 
let a = Matrix::<f32, 3, 3>::from( [ vec3( 1.0, 0.0, 0.0 ),
                                     vec3( 0.0, 1.0, 0.0 ),
                                     vec3( 0.0, 0.0, 1.0 ), ] );
let b: Matrix::<i32, 3, 3> =
            mat3x3( 0, -3, 5,
                    6, 1, -4,
                    2, 3, -2 );
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
let a: Matrix::<i32, 3, 3> =
    mat3x3( 0, -3, 5,
            6, 1, -4,
            2, 3, -2 );
let b: Matrix::<i32, 3, 3> =
    mat3x3( -1, 0, -3,
             4, 5, 1,
             2, 6, -2 );
let c: Matrix::<i32, 3, 3> =
    mat3x3( -2, 15, -13,
            -10, -19, -9,
             6, 3, 1 );
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
    mat3x3( 1i32, 0, 0,
            0, 2, 0,
            0, 0, 3 )
        .diagonal(),
    vec3( 1i32, 2, 3 ) 
);

assert_eq!(
    mat4x4( 1i32, 0, 0, 0, 
            0, 2, 0, 0, 
            0, 0, 3, 0, 
            0, 0, 0, 4 )
        .diagonal(),
    vec4( 1i32, 2, 3, 4 ) 
);
```          

## Provided traits

### `Zero` and `One`

Defines the additive and multiplicative identity respectively. `zero` returns 
vectors and matrices filled with zeroes, while `one` returns the unit vector
and unit matrices. 

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

## Limitations

Individual component access isn't implemented for vectors and most likely will 
not be as simple as a public field access when it is. For now, manually 
destructure the vector.

Truncation must be performed by manually destructuring as well, but this is due
to a limitation of the current compiler.

## Future work 

There is a lot of work that needs to be done on aljabar before it can be 
considered stable or non-experimental.

* Implement `Matrix::invert`.
* Add optional Serde support.
* Add and implement rotational and angular data structures. 
* Convert `Vector` ops to SIMD implementations.
* Add more tests and get code coverage to 100%.
* Implement better methods for accessing components of Vectors. 
* Add `Point` type.
* Implement faster matrix multiplication algorithms. 
* Add procedural macro to create matrices of any width and height combination
