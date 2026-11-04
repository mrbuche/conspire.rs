#[cfg(test)]
mod test;

use super::{expansion::Expansion, orient2d, orient3d};
use crate::geometry::Coordinate;
use std::cmp::Ordering;

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
pub fn along(
    p: &Coordinate<3>,
    q: &Coordinate<3>,
    x: &Cutter,
    y: &Cutter,
) -> Result<Ordering, &'static str> {
    let [x_at_p, x_at_q] = values(x, p, q)?;
    let [y_at_p, y_at_q] = values(y, p, q)?;
    let x_change = x_at_p.clone().minus(x_at_q.clone());
    let y_change = y_at_p.clone().minus(y_at_q.clone());
    if x_change.sign() == Ordering::Equal || y_change.sign() == Ordering::Equal {
        return Err("the line is parallel to a cutter");
    }
    let numerator = x_at_p.times(&y_at_q).minus(y_at_p.times(&x_at_q));
    let sign = numerator.sign() as i8 * x_change.sign() as i8 * y_change.sign() as i8;
    Ok(0.cmp(&sign))
}

/// The function of a cutter, which is zero where the line meets it, at `p` and at `q`.
fn values(
    cutter: &Cutter,
    p: &Coordinate<3>,
    q: &Coordinate<3>,
) -> Result<[Expansion; 2], &'static str> {
    match cutter {
        Cutter::Face([a, b, c]) => Ok([p, q].map(|point| orient3d::determinant(a, b, c, point))),
        Cutter::Edge([r, s]) => (0..3)
            .find_map(|axis| {
                let values = [p, q].map(|point| orient2d::determinant(r, s, point, axis));
                (values[0].clone().minus(values[1].clone()).sign() != Ordering::Equal)
                    .then_some(values)
            })
            .ok_or("the line is parallel to an edge it should cross"),
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
            Ok(
                [p, q]
                    .map(|point| Expansion::difference(point[axis].value(), vertex[axis].value())),
            )
        }
    }
}
