//! Constructive solid geometry: analytic primitives (and, later, their boolean
//! combinations), each meshable through [`Solid`](crate::geometry::solid::Solid).

mod cuboid;
mod cylinder;
pub mod ops;
mod primitive;
mod sphere;

pub use cuboid::{Cuboid, CuboidOracle};
pub use cylinder::{Cylinder, CylinderOracle};
pub use primitive::{Primitive, PrimitiveOracle};
pub use sphere::{Sphere, SphereOracle};
