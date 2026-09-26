#![cfg(all(feature = "fem", feature = "autodiff"))]

#[macro_use]
#[path = "../src/domain/fem/block/element/solid/hyperelastic/autodiff/test.rs"]
mod common;

#[macro_use]
#[path = "../src/domain/fem/block/element/solid/hyperviscoelastic/autodiff/test.rs"]
mod hyperviscoelastic;

#[path = "../src/domain/fem/block/element/linear/hexahedron/autodiff/test.rs"]
mod linear_hexahedron;
#[path = "../src/domain/fem/block/element/linear/tetrahedron/autodiff/test.rs"]
mod linear_tetrahedron;
#[path = "../src/domain/fem/block/element/planar/autodiff/test.rs"]
mod linear_triangle;
