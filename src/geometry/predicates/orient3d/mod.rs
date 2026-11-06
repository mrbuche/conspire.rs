#[cfg(test)]
mod test;

use super::{HALF_EPSILON, arithmetic::Arithmetic, expansion::Expansion};
use crate::geometry::Coordinate;
use core::{array::from_fn, cmp::Ordering};

/// A bound on the relative rounding error of the floating-point `orient3d` determinant.[^1]
///
/// [^1]: J. R. Shewchuk, [Discrete Comput. Geom. **18**, 305 (1997)](https://doi.org/10.1007/PL00009321).
const ERROR_BOUND: f64 = (7.0 + 56.0 * HALF_EPSILON) * HALF_EPSILON;

/// The sign of `(a - d) · ((b - d) × (c - d))`, exactly.
///
/// This is positive when `d` lies below the plane through `a`, `b` and `c`, with them seen
/// counterclockwise from above, negative when above, and zero when `d` is on the plane.
/// Overflow and underflow of the intermediate products are not handled.
pub fn orient3d(
    a: &Coordinate<3>,
    b: &Coordinate<3>,
    c: &Coordinate<3>,
    d: &Coordinate<3>,
) -> Ordering {
    let [a, b, c, d] = [a, b, c, d].map(|point| from_fn(|axis| point[axis].value()));
    filtered(a, b, c, d).unwrap_or_else(|| exact(a, b, c, d))
}

/// Whether three points lie on one line, exactly.
pub fn collinear(a: &Coordinate<3>, b: &Coordinate<3>, c: &Coordinate<3>) -> bool {
    let [a, b, c]: [[f64; 3]; 3] = [a, b, c].map(|point| from_fn(|axis| point[axis].value()));
    let [ux, uy, uz] = from_fn(|axis| Expansion::difference(b[axis], a[axis]));
    let [vx, vy, vz] = from_fn(|axis| Expansion::difference(c[axis], a[axis]));
    [
        uy.times(&vz).minus(uz.times(&vy)),
        uz.times(&vx).minus(ux.times(&vz)),
        ux.times(&vy).minus(uy.times(&vx)),
    ]
    .iter()
    .all(|component| component.sign() == Ordering::Equal)
}

/// The floating-point determinant behind [`orient3d`], proportional to the signed distance of `d`
/// from the plane through `a`, `b` and `c`.
///
/// It carries rounding error, so it can place a point on a plane but never decide a sign.
pub fn orient3d_estimate(
    a: &Coordinate<3>,
    b: &Coordinate<3>,
    c: &Coordinate<3>,
    d: &Coordinate<3>,
) -> f64 {
    let [a, b, c, d] = [a, b, c, d].map(|point| from_fn(|axis| point[axis].value()));
    estimate(a, b, c, d).0
}

/// The determinant, and the sum of the magnitudes of its terms, which bounds its rounding error.
fn estimate(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> (f64, f64) {
    let [adx, ady, adz] = from_fn(|axis| a[axis] - d[axis]);
    let [bdx, bdy, bdz] = from_fn(|axis| b[axis] - d[axis]);
    let [cdx, cdy, cdz] = from_fn(|axis| c[axis] - d[axis]);
    let (bdxcdy, cdxbdy) = (bdx * cdy, cdx * bdy);
    let (cdxady, adxcdy) = (cdx * ady, adx * cdy);
    let (adxbdy, bdxady) = (adx * bdy, bdx * ady);
    let determinant = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
    let permanent = (bdxcdy.abs() + cdxbdy.abs()) * adz.abs()
        + (cdxady.abs() + adxcdy.abs()) * bdz.abs()
        + (adxbdy.abs() + bdxady.abs()) * cdz.abs();
    (determinant, permanent)
}

/// The sign from floating point when it is certain, and `None` when rounding could change it.
fn filtered(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> Option<Ordering> {
    let (determinant, permanent) = estimate(a, b, c, d);
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
fn exact(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> Ordering {
    value::<Expansion>(a, b, c, d).sign()
}

/// The determinant in any arithmetic: exactly as an expansion, or as an estimate with its error.
pub(super) fn value<A: Arithmetic>(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> A {
    let [adx, ady, adz] = from_fn(|axis| A::difference(a[axis], d[axis]));
    let [bdx, bdy, bdz] = from_fn(|axis| A::difference(b[axis], d[axis]));
    let [cdx, cdy, cdz] = from_fn(|axis| A::difference(c[axis], d[axis]));
    let cross_z = bdx.times(&cdy).minus(cdx.times(&bdy));
    let cross_x = cdx.times(&ady).minus(adx.times(&cdy));
    let cross_y = adx.times(&bdy).minus(bdx.times(&ady));
    adz.times(&cross_z)
        .sum(bdz.times(&cross_x))
        .sum(cdz.times(&cross_y))
}
