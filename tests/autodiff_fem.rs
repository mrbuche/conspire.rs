#![cfg(all(feature = "fem", feature = "autodiff"))]

#[macro_use]
#[path = "../src/domain/fem/block/element/solid/hyperelastic/autodiff/test.rs"]
mod common;

#[macro_use]
#[path = "../src/domain/fem/block/element/solid/hyperviscoelastic/autodiff/test.rs"]
mod hyperviscoelastic;

#[macro_use]
#[path = "../src/domain/fem/block/element/planar/autodiff/test.rs"]
mod planar;

#[path = "../src/domain/fem/block/element/linear/hexahedron/autodiff/test.rs"]
mod linear_hexahedron;

#[path = "../src/domain/fem/block/element/linear/tetrahedron/autodiff/test.rs"]
mod linear_tetrahedron;

#[path = "../src/domain/fem/block/element/linear/pyramid/autodiff/test.rs"]
mod linear_pyramid;

#[path = "../src/domain/fem/block/element/linear/wedge/autodiff/test.rs"]
mod linear_wedge;

#[path = "../src/domain/fem/block/element/quadratic/hexahedron/autodiff/test.rs"]
mod quadratic_hexahedron;

#[path = "../src/domain/fem/block/element/quadratic/tetrahedron/autodiff/test.rs"]
mod quadratic_tetrahedron;

#[path = "../src/domain/fem/block/element/quadratic/pyramid/autodiff/test.rs"]
mod quadratic_pyramid;

#[path = "../src/domain/fem/block/element/quadratic/wedge/autodiff/test.rs"]
mod quadratic_wedge;

#[path = "../src/domain/fem/block/element/serendipity/hexahedron/autodiff/test.rs"]
mod serendipity_hexahedron;

#[path = "../src/domain/fem/block/element/planar/triangle/autodiff/test.rs"]
mod linear_triangle;

#[path = "../src/domain/fem/block/element/planar/quadrilateral/autodiff/test.rs"]
mod linear_quadrilateral;
