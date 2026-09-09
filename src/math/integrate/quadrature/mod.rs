//! Fixed-rule numerical quadrature.
//!
//! Gauss rules are built by the Golub-Welsch algorithm: the nodes are the
//! eigenvalues of the Jacobi matrix of the associated orthogonal polynomials
//! and the weights come from the first components of its eigenvectors. Nodes
//! and weights are computed on demand; callers that need them in a hot loop
//! should compute once and cache.

#[cfg(test)]
mod test;

mod gauss;
mod sphere;

pub use gauss::{gauss_hermite, gauss_laguerre, gauss_legendre};
pub use sphere::{SphereNode, sphere_product};
