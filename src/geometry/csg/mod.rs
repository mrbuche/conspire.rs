//! Constructive solid geometry: analytic primitives (and, later, their boolean
//! combinations), each meshable through [`Solid`](crate::geometry::solid::Solid).

mod cuboid;

pub use cuboid::{Cuboid, CuboidOracle};
