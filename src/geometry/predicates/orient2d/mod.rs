#[cfg(test)]
mod test;

use super::{HALF_EPSILON, arithmetic::Arithmetic, expansion::Expansion, project};
use crate::geometry::Coordinate;
use core::{array::from_fn, cmp::Ordering};

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

/// The sign of the orientation of `a`, `b` and `c` in a plane given by two coordinates, exactly.
///
/// This is positive when they run counterclockwise, negative when clockwise and zero when they
/// are collinear: [`orient2d`] for points that are already flat.
pub(crate) fn orient2d_planar(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Ordering {
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
    value::<Expansion>(a, b, c).sign()
}

/// The determinant in any arithmetic: exactly as an expansion, or as an estimate with its error.
pub(super) fn value<A: Arithmetic>(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> A {
    let [acx, acy] = from_fn(|index| A::difference(a[index], c[index]));
    let [bcx, bcy] = from_fn(|index| A::difference(b[index], c[index]));
    acx.times(&bcy).minus(acy.times(&bcx))
}
