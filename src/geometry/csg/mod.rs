//! Solids known through their signed distance: analytic primitives and their boolean
//! combinations.

#[cfg(test)]
mod contract;
mod cuboid;
pub mod ops;
mod primitive;
mod sphere;

pub use cuboid::Cuboid;
pub use primitive::Primitive;
pub use sphere::Sphere;
