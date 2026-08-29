//! Constructive solid geometry: analytic primitives (and, later, their boolean
//! combinations), each meshable through [`Solid`](crate::geometry::solid::Solid).

mod cone;
mod cuboid;
mod cylinder;
mod ellipsoid;
pub mod ops;
mod primitive;
mod sphere;
mod torus;

pub use cone::{Cone, ConeOracle};
pub use cuboid::{Cuboid, CuboidOracle};
pub use cylinder::{Cylinder, CylinderOracle};
pub use ellipsoid::{Ellipsoid, EllipsoidOracle};
pub use primitive::{Primitive, PrimitiveOracle};
pub use sphere::{Sphere, SphereOracle};
pub use torus::{Torus, TorusOracle};
