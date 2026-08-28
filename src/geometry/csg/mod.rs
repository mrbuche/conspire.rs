//! Constructive solid geometry: analytic primitives (and, later, their boolean
//! combinations), each meshable through [`Solid`](crate::geometry::solid::Solid).

mod cuboid;
mod cylinder;
mod sphere;

pub use cuboid::{Cuboid, CuboidOracle};
pub use cylinder::{Cylinder, CylinderOracle};
pub use sphere::{Sphere, SphereOracle};
