//! Solids known through their signed distance: analytic primitives and their boolean
//! combinations.

mod axis;
mod cone;
#[cfg(test)]
mod contract;
mod cuboid;
mod cylinder;
mod ellipsoid;
pub mod ops;
mod primitive;
mod sphere;
mod torus;

pub use cone::Cone;
pub use cuboid::Cuboid;
pub use cylinder::Cylinder;
pub use ellipsoid::Ellipsoid;
pub use primitive::Primitive;
pub use sphere::Sphere;
pub use torus::Torus;
