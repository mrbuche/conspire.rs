#[cfg(test)]
mod test;

use super::{HALF_EPSILON, expansion::Expansion};
use crate::geometry::Coordinate;
use std::{array::from_fn, cmp::Ordering};

/// A bound on the relative rounding error of the floating-point `orient2d` determinant.[^1]
///
/// [^1]: J. R. Shewchuk, [Discrete Comput. Geom. **18**, 305 (1997)](https://doi.org/10.1007/PL00009321).
const ERROR_BOUND: f64 = (3.0 + 16.0 * HALF_EPSILON) * HALF_EPSILON;

/// The sign of the orientation of `a`, `b` and `c` in the plane, after dropping coordinate `axis`.
///
/// This is positive when they run counterclockwise in the remaining two coordinates, taken in
/// cyclic order, so for `axis` 2 it is counterclockwise seen from above. For points in one plane,
/// dropping an axis along which its normal has a component keeps their orientation, up to the
/// sign of that component, so this decides in-plane questions exactly.
pub fn orient2d(a: &Coordinate<3>, b: &Coordinate<3>, c: &Coordinate<3>, axis: usize) -> Ordering {
    let [a, b, c] = project([a, b, c], axis);
    filtered(a, b, c).unwrap_or_else(|| exact(a, b, c))
}

/// The floating-point determinant behind [`orient2d`], with rounding error: it can place a point
/// on a line but never decide a sign.
pub fn orient2d_estimate(
    a: &Coordinate<3>,
    b: &Coordinate<3>,
    c: &Coordinate<3>,
    axis: usize,
) -> f64 {
    let [a, b, c] = project([a, b, c], axis);
    estimate(a, b, c).0
}

/// The coordinates left after dropping `axis`, in cyclic order.
fn project(points: [&Coordinate<3>; 3], axis: usize) -> [[f64; 2]; 3] {
    let (first, second) = ((axis + 1) % 3, (axis + 2) % 3);
    points.map(|point| [point[first].value(), point[second].value()])
}

/// The determinant, and the sum of the magnitudes of its terms, which bounds its rounding error.
fn estimate(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> (f64, f64) {
    let (left, right) = ((a[0] - c[0]) * (b[1] - c[1]), (a[1] - c[1]) * (b[0] - c[0]));
    (left - right, left.abs() + right.abs())
}

/// The sign from floating point when it is certain, and `None` when rounding could change it.
fn filtered(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<Ordering> {
    let (determinant, permanent) = estimate(a, b, c);
    let bound = ERROR_BOUND * permanent;
    if determinant > bound {
        Some(Ordering::Greater)
    } else if -determinant > bound {
        Some(Ordering::Less)
    } else {
        None
    }
}

/// The sign from expansion arithmetic, which is exact.
fn exact(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Ordering {
    let [acx, acy] = from_fn(|index| Expansion::difference(a[index], c[index]));
    let [bcx, bcy] = from_fn(|index| Expansion::difference(b[index], c[index]));
    acx.times(&bcy).minus(acy.times(&bcx)).sign()
}
