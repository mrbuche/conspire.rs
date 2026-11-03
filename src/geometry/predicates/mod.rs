//! Exact geometric predicates, which never return the wrong sign from rounding.

#[cfg(test)]
mod test;

mod expansion;
mod orient2d;
mod orient3d;

pub use orient2d::{orient2d, orient2d_estimate};
pub use orient3d::{collinear, orient3d, orient3d_estimate};

/// Half the spacing of the floats near one, which the bounds on rounding error are multiples of.
const HALF_EPSILON: f64 = f64::EPSILON / 2.0;
