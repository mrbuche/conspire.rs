#[cfg(test)]
mod test;

use super::{
    arithmetic::{Arithmetic, Estimate},
    expansion::Expansion,
    orient2d, orient3d, project,
};
use crate::geometry::Coordinate;
use core::{array::from_fn, cmp::Ordering};

/// What a line meets at one point: a triangle, an edge or a vertex.
pub enum Cutter<'a> {
    /// The plane of a triangle.
    Face([&'a Coordinate<3>; 3]),
    /// The line through an edge, which the line must cross.
    Edge([&'a Coordinate<3>; 2]),
    /// A vertex on the line.
    Vertex(&'a Coordinate<3>),
}

/// Which of two points on the line from `p` to `q` comes first, exactly, with each point given as
/// where the line meets a [`Cutter`]: `Less` when `x` is nearer `p` than `y` is, and `Equal` when
/// they are the same point.
///
/// This decides the order of points along an edge from the original vertices alone, however their
/// coordinates would be rounded. The points of a pair of cutters need not be of the same kind. A
/// cutter the line is parallel to has no point to compare, and is an error, as is `p = q`.
///
/// Each cutter gives a function that is affine along the line and zero at its point, `f`. A point
/// is at `p + t (q - p)` with `t = f(p) / (f(p) - f(q))`, so the order of two points is the sign of
/// `f_x(p) f_y(q) - f_y(p) f_x(q)`, corrected by the signs of `f(p) - f(q)` of each. The functions
/// are determinants of the original coordinates, and the comparison is of degree at most 6.
///
/// The sign is first found in floating point, with a bound on its rounding error, and in exact
/// arithmetic only when that bound does not settle it.
pub fn along(
    p: &Coordinate<3>,
    q: &Coordinate<3>,
    x: &Cutter,
    y: &Cutter,
) -> Result<Ordering, &'static str> {
    if let Some(found) = decide::<Estimate>(p, q, x, y)? {
        return Ok(found);
    }
    decide::<Expansion>(p, q, x, y)?.ok_or("exact arithmetic is always certain")
}

/// The order, from the sign of the comparison in some arithmetic, or `None` when that arithmetic
/// cannot be certain of it.
fn decide<A: Arithmetic>(
    p: &Coordinate<3>,
    q: &Coordinate<3>,
    x: &Cutter,
    y: &Cutter,
) -> Result<Option<Ordering>, &'static str> {
    let Some([x_at_p, x_at_q]) = values::<A>(x, p, q)? else {
        return Ok(None);
    };
    let Some([y_at_p, y_at_q]) = values::<A>(y, p, q)? else {
        return Ok(None);
    };
    let x_change = x_at_p.clone().minus(x_at_q.clone());
    let y_change = y_at_p.clone().minus(y_at_q.clone());
    let numerator = x_at_p.times(&y_at_q).minus(y_at_p.times(&x_at_q));
    let (Some(x_change), Some(y_change), Some(numerator)) = (
        x_change.certain_sign(),
        y_change.certain_sign(),
        numerator.certain_sign(),
    ) else {
        return Ok(None);
    };
    if x_change == Ordering::Equal || y_change == Ordering::Equal {
        return Err("the line is parallel to a cutter");
    }
    let sign = numerator as i8 * x_change as i8 * y_change as i8;
    Ok(Some(0.cmp(&sign)))
}

/// The function of a cutter, which is zero where the line meets it, at `p` and at `q`, or `None`
/// when the arithmetic cannot be certain which of the functions to use.
fn values<A: Arithmetic>(
    cutter: &Cutter,
    p: &Coordinate<3>,
    q: &Coordinate<3>,
) -> Result<Option<[A; 2]>, &'static str> {
    let raw = |point: &Coordinate<3>| -> [f64; 3] { from_fn(|axis| point[axis].value()) };
    match cutter {
        Cutter::Face([a, b, c]) => {
            let [a, b, c] = [a, b, c].map(|point| raw(point));
            Ok(Some(
                [p, q].map(|point| orient3d::value::<A>(a, b, c, raw(point))),
            ))
        }
        Cutter::Edge([r, s]) => {
            for axis in 0..3 {
                let [r, s, p, q] = project([r, s, p, q], axis);
                let values = [p, q].map(|point| orient2d::value::<A>(r, s, point));
                match values[0].clone().minus(values[1].clone()).certain_sign() {
                    Some(Ordering::Equal) => continue,
                    Some(_) => return Ok(Some(values)),
                    None => return Ok(None),
                }
            }
            Err("the line is parallel to an edge it should cross")
        }
        Cutter::Vertex(vertex) => {
            let axis = (0..3)
                .max_by(|&a, &b| {
                    (q[a].value() - p[a].value())
                        .abs()
                        .total_cmp(&(q[b].value() - p[b].value()).abs())
                })
                .unwrap_or(0);
            if p[axis] == q[axis] {
                return Err("the line has no direction");
            }
            Ok(Some([p, q].map(|point| {
                A::difference(point[axis].value(), vertex[axis].value())
            })))
        }
    }
}
