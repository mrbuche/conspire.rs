//! Exact geometric predicates, which never return the wrong sign from rounding.

#[cfg(test)]
mod test;

mod expansion;
mod incircle;
mod orient2d;
mod orient3d;

pub use incircle::incircle;
pub use orient2d::{orient2d, orient2d_estimate};
pub use orient3d::{collinear, orient3d, orient3d_estimate};

use crate::geometry::Coordinate;

/// Half the spacing of the floats near one, which the bounds on rounding error are multiples of.
const HALF_EPSILON: f64 = f64::EPSILON / 2.0;

/// The coordinates left after dropping `axis`, in cyclic order.
fn project<const N: usize>(points: [&Coordinate<3>; N], axis: usize) -> [[f64; 2]; N] {
    let (first, second) = ((axis + 1) % 3, (axis + 2) % 3);
    points.map(|point| [point[first].value(), point[second].value()])
}
