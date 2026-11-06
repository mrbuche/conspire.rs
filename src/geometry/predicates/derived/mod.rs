//! Points that are not given but are where given things meet, and which side of a plane they are on.
//!
//! A point where a line meets a plane, or where three planes meet, is a rational function of the
//! coordinates of the points that define them, and rounding it to coordinates loses the answer to
//! questions such as which side of another plane it is on. These are decided from the defining
//! points instead, so the answer is exact however close to the plane the point is.

#[cfg(test)]
mod test;

use super::{
    arithmetic::{Arithmetic, Estimate},
    expansion::Expansion,
    orient3d::{self, orient3d},
};
use crate::geometry::Coordinate;
use core::{array::from_fn, cmp::Ordering};

/// A point defined by points that are given.
pub enum Derived<'a> {
    /// A given point.
    Vertex(&'a Coordinate<3>),
    /// Where the line through two points meets the plane through three.
    Lpi {
        line: [&'a Coordinate<3>; 2],
        plane: [&'a Coordinate<3>; 3],
    },
    /// Where three planes meet, each through three points.
    Tpi([[&'a Coordinate<3>; 3]; 3]),
}

type Point = [f64; 3];

fn raw(point: &Coordinate<3>) -> Point {
    from_fn(|axis| point[axis].value())
}

/// On which side of the plane through `plane` the point is, exactly: what [`orient3d`] says of a
/// point that is given, which is positive below the plane, with the three points counterclockwise
/// seen from above.
///
/// It fails where the point is not one: a line parallel to its plane, or three planes that do not
/// meet in one point.
///
/// For the point where a line through `p` and `q` meets the plane `u`, this is the sign of
/// `F_u(p) F_v(q) - F_v(p) F_u(q)` over `F_u(p) - F_u(q)`, where `F` is `orient3d` with the plane
/// as its first three points, which is degree 6 in the coordinates. For three planes it is the sign
/// of `-n_v · X / w`, with `n` the normals, `w` their triple product and `X` the sum of
/// `n_j · (a_j - o)` times the cross product of the other two normals, which is degree 9. The
/// sign is found in floating point with a bound on its rounding error, and exactly only when that
/// bound does not settle it.
pub fn side(plane: [&Coordinate<3>; 3], point: &Derived) -> Result<Ordering, &'static str> {
    if let Derived::Vertex(vertex) = point {
        return Ok(orient3d(plane[0], plane[1], plane[2], vertex));
    }
    let plane = plane.map(raw);
    if let Some(found) = decide::<Estimate>(plane, point)? {
        return Ok(found);
    }
    decide::<Expansion>(plane, point)?.ok_or("exact arithmetic is always certain")
}

impl Derived<'_> {
    /// Where the point is, rounded to coordinates: to be placed, never to be decided from.
    pub fn position(&self) -> Result<Coordinate<3>, &'static str> {
        match self {
            Derived::Vertex(vertex) => Ok((*vertex).clone()),
            Derived::Lpi { line, plane } => {
                let [p, q] = line.map(raw);
                let plane = plane.map(raw);
                let (a, b) = plane_values::<Expansion>(plane, p, q);
                let denominator = a.clone().minus(b).estimate();
                if denominator == 0.0 {
                    return Err("the line does not meet the plane at a point");
                }
                let fraction = a.estimate() / denominator;
                Ok(from_fn(|axis| p[axis] + fraction * (q[axis] - p[axis])).into())
            }
            Derived::Tpi(planes) => {
                let planes = planes.map(|plane| plane.map(raw));
                let origin = planes[0][0];
                let (x, w) = triple::<Expansion>(planes, origin);
                let w = w.estimate();
                if w == 0.0 {
                    return Err("the planes do not meet at a point");
                }
                Ok(from_fn(|axis| origin[axis] + x[axis].estimate() / w).into())
            }
        }
    }
}

/// The sign of the point's side of the plane in some arithmetic, or `None` when that arithmetic
/// cannot be certain of it.
fn decide<A: Arithmetic>(
    plane: [Point; 3],
    point: &Derived,
) -> Result<Option<Ordering>, &'static str> {
    match point {
        Derived::Vertex(vertex) => {
            Ok(orient3d::value::<A>(plane[0], plane[1], plane[2], raw(vertex)).certain_sign())
        }
        Derived::Lpi { line, plane: u } => {
            let [p, q] = line.map(raw);
            let u = u.map(raw);
            // `F_u` of the ends, and `F_v` of them, with `v` the plane that is being tested.
            let (a, b) = plane_values::<A>(u, p, q);
            let (c, d) = plane_values::<A>(plane, p, q);
            let numerator = a.times(&d).minus(c.times(&b));
            let denominator = a.minus(b);
            let (Some(numerator), Some(denominator)) =
                (numerator.certain_sign(), denominator.certain_sign())
            else {
                return Ok(None);
            };
            if denominator == Ordering::Equal {
                return Err("the line does not meet the plane at a point");
            }
            Ok(Some((numerator as i8 * denominator as i8).cmp(&0)))
        }
        Derived::Tpi(planes) => {
            let planes = planes.map(|plane| plane.map(raw));
            let origin = plane[0];
            let (x, w) = triple::<A>(planes, origin);
            let normal = normal::<A>(plane);
            let numerator = dot(&normal, &x);
            let (Some(numerator), Some(w)) = (numerator.certain_sign(), w.certain_sign()) else {
                return Ok(None);
            };
            if w == Ordering::Equal {
                return Err("the planes do not meet at a point");
            }
            Ok(Some((-(numerator as i8 * w as i8)).cmp(&0)))
        }
    }
}

/// `orient3d` of the plane at two points.
fn plane_values<A: Arithmetic>(plane: [Point; 3], p: Point, q: Point) -> (A, A) {
    (
        orient3d::value::<A>(plane[0], plane[1], plane[2], p).compact(),
        orient3d::value::<A>(plane[0], plane[1], plane[2], q).compact(),
    )
}

/// Where three planes meet, from the point `origin`: the numerator `X` of the position, which is
/// `X / w`, and the determinant `w`.
fn triple<A: Arithmetic>(planes: [[Point; 3]; 3], origin: Point) -> ([A; 3], A) {
    let normals: [[A; 3]; 3] = planes.map(normal::<A>);
    // The plane `i` is `n_i · x = d_i`, for `x` measured from the origin.
    let offsets: [A; 3] =
        from_fn(|i| dot(&normals[i], &difference::<A>(planes[i][0], origin)).compact());
    let crosses = [
        cross(&normals[1], &normals[2]),
        cross(&normals[2], &normals[0]),
        cross(&normals[0], &normals[1]),
    ];
    let determinant = dot(&normals[0], &crosses[0]).compact();
    let numerator = from_fn(|axis| {
        offsets[0]
            .times(&crosses[0][axis])
            .sum(offsets[1].times(&crosses[1][axis]))
            .sum(offsets[2].times(&crosses[2][axis]))
            .compact()
    });
    (numerator, determinant)
}

fn difference<A: Arithmetic>(a: Point, b: Point) -> [A; 3] {
    from_fn(|axis| A::difference(a[axis], b[axis]))
}

/// The normal of the plane through three points, as the cross product of two of its edges.
fn normal<A: Arithmetic>([a, b, c]: [Point; 3]) -> [A; 3] {
    cross(&difference::<A>(b, a), &difference::<A>(c, a))
}

fn cross<A: Arithmetic>(u: &[A; 3], v: &[A; 3]) -> [A; 3] {
    [
        u[1].times(&v[2]).minus(u[2].times(&v[1])).compact(),
        u[2].times(&v[0]).minus(u[0].times(&v[2])).compact(),
        u[0].times(&v[1]).minus(u[1].times(&v[0])).compact(),
    ]
}

fn dot<A: Arithmetic>(u: &[A; 3], v: &[A; 3]) -> A {
    u[0].times(&v[0])
        .sum(u[1].times(&v[1]))
        .sum(u[2].times(&v[2]))
}
