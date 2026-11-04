//! Solids known through their signed distance: analytic primitives and their boolean
//! combinations.

mod axis;
mod cone;
#[cfg(test)]
mod contract;
mod cuboid;
mod cylinder;
pub mod ops;
mod primitive;
mod sphere;

pub use cone::Cone;
pub use cuboid::Cuboid;
pub use cylinder::Cylinder;
pub use primitive::Primitive;
pub use sphere::Sphere;
