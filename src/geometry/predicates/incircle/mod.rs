#[cfg(test)]
mod test;

use super::{HALF_EPSILON, expansion::Expansion, project};
use crate::geometry::Coordinate;
use core::{array::from_fn, cmp::Ordering};

/// A bound on the relative rounding error of the floating-point `incircle` determinant, as a
/// multiple of its permanent: the sum of the magnitudes of its terms.
///
/// This is derived, not tested: random tests reach nowhere near the worst case, so they cannot
/// tell a sufficient bound from an insufficient one. Take every floating-point operation to be
/// correctly rounded with relative error at most `ε`, with no overflow or underflow, and count in
/// units of `ε`, to first order.
/// - A difference of coordinates has error 1.
/// - A term of a lift, such as `g₀·x·x`, uses its difference twice and is then multiplied twice,
///   so it has error `2 + 2 = 4`; and the cross term `2·g₁·x·y` is the same, since doubling is
///   exact. The two sums that combine the three terms add 2 more, so a lift has error 6 relative
///   to the sum of the magnitudes of its terms.
/// - A minor `p·q − r·s` uses four differences once each and one product each, so each product has
///   error `2 + 1 = 3`, and the subtraction adds 1, giving 4 relative to `|pq| + |rs|`.
/// - A lift times a minor has error `6 + 4` from its factors, since the error of a product is the
///   sum of the relative errors of its factors, plus 1 for its own rounding: 11.
/// - The two sums that combine the three products add 2, giving 13.
///
/// So the error is at most `13ε` times the permanent, plus terms of order `ε²`. The constant `14`
/// leaves a margin of `ε`, which exceeds the second-order terms, about `170ε²`, by a factor of
/// about `10¹⁴`; it also covers the rounding of the permanent itself.
///
/// Shewchuk's bound for the Euclidean determinant is `(10 + 96ε)ε`,[^1] from a finer analysis than
/// this count, which would give 11 for the Euclidean lift `x² + y²`. This one is not tight. A bound
/// that is too large only sends more cases to the exact path; one that is too small would be
/// wrong.
///
/// [^1]: J. R. Shewchuk, [Discrete Comput. Geom. **18**, 305 (1997)](https://doi.org/10.1007/PL00009321).
const ERROR_BOUND: f64 = (14.0 + 128.0 * HALF_EPSILON) * HALF_EPSILON;

/// The sign of the in-circle determinant of `a`, `b`, `c` and `d` in the plane, after dropping
/// coordinate `axis`, exactly.
///
/// With `a`, `b` and `c` counterclockwise in the remaining two coordinates, taken in cyclic order
/// as for [`orient2d`](super::orient2d), this is positive when `d` is inside the circle through
/// them, negative when outside, and zero when on it. Overflow and underflow of the intermediate
/// products are not handled.
pub fn incircle(
    a: &Coordinate<3>,
    b: &Coordinate<3>,
    c: &Coordinate<3>,
    d: &Coordinate<3>,
    axis: usize,
) -> Ordering {
    let [a, b, c, d] = project([a, b, c, d], axis);
    incircle_metric(a, b, c, d, [1.0, 0.0, 1.0])
}

/// The sign of the in-circle determinant for lengths measured by `x² g₀ + 2xy g₁ + y² g₂`, with
/// `gram` as `[g₀, g₁, g₂]`.
///
/// This is [`incircle`] after a linear change of coordinates, as for a triangle whose coordinates
/// are along two of its edges: the circles are those of the true shape of the triangle. With
/// `gram` as `[1, 0, 1]` it is the ordinary in-circle test, and the form should be positive
/// definite.
pub(crate) fn incircle_metric(
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
    gram: [f64; 3],
) -> Ordering {
    filtered(a, b, c, d, gram).unwrap_or_else(|| exact(a, b, c, d, gram))
}

/// The determinant, and the sum of the magnitudes of its terms, which bounds its rounding error.
///
/// The lift of a point is its squared length from `d`. The linear part of a general circle's lift
/// cancels against the first two columns, so no centre is needed.
fn estimate(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2], gram: [f64; 3]) -> (f64, f64) {
    let [adx, ady] = from_fn(|axis| a[axis] - d[axis]);
    let [bdx, bdy] = from_fn(|axis| b[axis] - d[axis]);
    let [cdx, cdy] = from_fn(|axis| c[axis] - d[axis]);
    let lift = |x: f64, y: f64| gram[0] * x * x + 2.0 * gram[1] * x * y + gram[2] * y * y;
    let size = |x: f64, y: f64| {
        gram[0].abs() * x * x + 2.0 * gram[1].abs() * (x * y).abs() + gram[2].abs() * y * y
    };
    let (bdxcdy, cdxbdy) = (bdx * cdy, cdx * bdy);
    let (cdxady, adxcdy) = (cdx * ady, adx * cdy);
    let (adxbdy, bdxady) = (adx * bdy, bdx * ady);
    let determinant = lift(adx, ady) * (bdxcdy - cdxbdy)
        + lift(bdx, bdy) * (cdxady - adxcdy)
        + lift(cdx, cdy) * (adxbdy - bdxady);
    let permanent = (bdxcdy.abs() + cdxbdy.abs()) * size(adx, ady)
        + (cdxady.abs() + adxcdy.abs()) * size(bdx, bdy)
        + (adxbdy.abs() + bdxady.abs()) * size(cdx, cdy);
    (determinant, permanent)
}

/// The sign from floating point when it is certain, and `None` when rounding could change it.
fn filtered(
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
    gram: [f64; 3],
) -> Option<Ordering> {
    let (determinant, permanent) = estimate(a, b, c, d, gram);
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
fn exact(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2], gram: [f64; 3]) -> Ordering {
    let [adx, ady] = from_fn(|axis| Expansion::difference(a[axis], d[axis]));
    let [bdx, bdy] = from_fn(|axis| Expansion::difference(b[axis], d[axis]));
    let [cdx, cdy] = from_fn(|axis| Expansion::difference(c[axis], d[axis]));
    let [g0, g1, g2] = [gram[0], 2.0 * gram[1], gram[2]].map(Expansion::new);
    let lift = |x: &Expansion, y: &Expansion| {
        g0.times(&x.times(x))
            .sum(g1.times(&x.times(y)))
            .sum(g2.times(&y.times(y)))
    };
    let minor_a = bdx.times(&cdy).minus(cdx.times(&bdy));
    let minor_b = cdx.times(&ady).minus(adx.times(&cdy));
    let minor_c = adx.times(&bdy).minus(bdx.times(&ady));
    lift(&adx, &ady)
        .times(&minor_a)
        .sum(lift(&bdx, &bdy).times(&minor_b))
        .sum(lift(&cdx, &cdy).times(&minor_c))
        .sign()
}
