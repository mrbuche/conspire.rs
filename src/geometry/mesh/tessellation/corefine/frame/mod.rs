//! The plane of a triangle as a coordinate frame for its triangulation.
//!
//! A triangle with corners `a`, `b` and `c` becomes the triangle with corners `(0, 0)`, `(1, 0)`
//! and `(0, 1)`, with circles measured by the metric of the plane it lies in. The points on its
//! edges are placed exactly on the edges of the frame, and in the exact order along the edge that
//! the features that cut it dictate, so that the two triangles that share an edge agree on its
//! subdivision whatever the rounded coordinates of its points are. Points inside are placed by
//! their rounded coordinates, and fail the frame if rounding puts them out of the triangle.

#[cfg(test)]
mod test;

use super::{Feature, Key, Side, triangulate::Input};
use crate::geometry::{
    Coordinate,
    predicates::{Cutter, along, orient2d_planar},
};
use std::{cmp::Ordering, collections::BTreeMap};

/// Edge points are placed at multiples of this on the edge from `(1, 0)` to `(0, 1)`, so that
/// both of their coordinates, which sum to one, are exact.
const GRID: f64 = (1u64 << 52) as f64;

/// A triangle in its frame, with the name of every point that is not a corner.
pub(super) struct Frame {
    pub input: Input,
    /// The key of each point after the three corners, in the order of the points.
    pub keys: Vec<Key>,
}

enum Place {
    Corner(usize),
    Edge(usize),
    Inside,
}

/// The frame of a triangle of `side`, with corners `triangle` named by `nodes`, for the points
/// that lie on it and the segments between them.
///
/// Every point has a key, whose feature on this tessellation says where it is on the triangle and
/// whose other feature says what cuts it, as a [`Cutter`] from `cutter`. A point on a corner
/// need not be listed. The ranks of the points are the order of their keys.
pub(super) fn frame<'a>(
    triangle: [&Coordinate<3>; 3],
    nodes: [usize; 3],
    side: Side,
    points: &[(Key, Coordinate<3>)],
    segments: &[[Key; 2]],
    cutter: impl Fn(&Feature) -> Cutter<'a>,
) -> Result<Frame, &'static str> {
    let corners = triangle.map(values);
    let first = difference(corners[1], corners[0]);
    let second = difference(corners[2], corners[0]);
    let gram = [dot(first, first), dot(first, second), dot(second, second)];
    let determinant = gram[0] * gram[2] - gram[1] * gram[1];
    if determinant.partial_cmp(&0.0) != Some(Ordering::Greater) || !determinant.is_finite() {
        return Err("the triangle is degenerate");
    }
    let place = |key: &Key| place(own(key, side), &nodes);
    let mut found: BTreeMap<&Key, &Coordinate<3>> = BTreeMap::new();
    for (key, position) in points {
        if !matches!(place(key)?, Place::Corner(_)) {
            found.insert(key, position);
        }
    }
    let keys: Vec<Key> = found.keys().map(|&key| key.clone()).collect();
    let mut sides: [Vec<usize>; 3] = Default::default();
    for (index, key) in keys.iter().enumerate() {
        if let Place::Edge(edge) = place(key)? {
            sides[edge].push(index + 3);
        }
    }
    let mut frame_points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    frame_points.resize(3 + keys.len(), [0.0, 0.0]);
    for (index, key) in keys.iter().enumerate() {
        if matches!(place(key)?, Place::Inside) {
            let position = difference(values(found[key]), corners[0]);
            let (a, b) = (dot(first, position), dot(second, position));
            let point = [
                (a * gram[2] - b * gram[1]) / determinant,
                (b * gram[0] - a * gram[1]) / determinant,
            ];
            let inside = [
                orient2d_planar([0.0, 0.0], [1.0, 0.0], point),
                orient2d_planar([1.0, 0.0], [0.0, 1.0], point),
                orient2d_planar([0.0, 1.0], [0.0, 0.0], point),
            ];
            if inside.iter().any(|&sign| sign != Ordering::Greater) {
                return Err("a point inside the triangle is rounded out of it");
            }
            frame_points[index + 3] = point;
        }
    }
    for (edge, indices) in sides.iter_mut().enumerate() {
        let (p, q) = (triangle[edge], triangle[(edge + 1) % 3]);
        order(indices, |&a, &b| {
            let cutter_of = |index: usize| cutter(other(&keys[index - 3], side));
            along(p, q, &cutter_of(a), &cutter_of(b))
        })?;
        let ends = [values(p), values(q)];
        let length = difference(ends[1], ends[0]);
        let approximate: Vec<f64> = indices
            .iter()
            .map(|&index| {
                let from_start = difference(values(found[&keys[index - 3]]), ends[0]);
                dot(from_start, length) / dot(length, length)
            })
            .collect();
        let placed = match edge {
            0 => increasing(&approximate, false)?,
            1 => increasing(&approximate, true)?,
            _ => {
                let from_corner: Vec<f64> = approximate.iter().rev().map(|t| 1.0 - t).collect();
                let mut placed = increasing(&from_corner, false)?;
                placed.reverse();
                placed
            }
        };
        for (&index, &t) in indices.iter().zip(&placed) {
            frame_points[index] = match edge {
                0 => [t, 0.0],
                1 => [1.0 - t, t],
                _ => [0.0, t],
            };
        }
    }
    let locate = |key: &Key| -> Result<usize, &'static str> {
        match place(key)? {
            Place::Corner(corner) => Ok(corner),
            _ => keys
                .binary_search(key)
                .map(|index| index + 3)
                .map_err(|_| "a segment ends at a point that was not given"),
        }
    };
    let segments = segments
        .iter()
        .map(|[a, b]| Ok([locate(a)?, locate(b)?]))
        .collect::<Result<Vec<_>, &'static str>>()?;
    Ok(Frame {
        input: Input {
            gram,
            points: frame_points,
            sides,
            segments,
        },
        keys,
    })
}

fn values(point: &Coordinate<3>) -> [f64; 3] {
    [0, 1, 2].map(|axis| point[axis].value())
}

fn difference(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|axis| a[axis] - b[axis])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|axis| a[axis] * b[axis]).sum()
}

/// The feature of a key on the tessellation of a triangle.
fn own(key: &Key, side: Side) -> &Feature {
    match side {
        Side::First => &key.first,
        Side::Second => &key.second,
    }
}

/// The feature of a key on the other tessellation, which cuts the triangle there.
fn other(key: &Key, side: Side) -> &Feature {
    match side {
        Side::First => &key.second,
        Side::Second => &key.first,
    }
}

fn place(feature: &Feature, nodes: &[usize; 3]) -> Result<Place, &'static str> {
    match feature {
        Feature::Vertex(node) => nodes
            .iter()
            .position(|n| n == node)
            .map(Place::Corner)
            .ok_or("a point is on a vertex that is not a corner of the triangle"),
        Feature::Edge([a, b]) => (0..3)
            .find(|&edge| {
                let (m, n) = (nodes[edge], nodes[(edge + 1) % 3]);
                [m.min(n), m.max(n)] == [*a, *b]
            })
            .map(Place::Edge)
            .ok_or("a point is on an edge that is not an edge of the triangle"),
        Feature::Face(_) => Ok(Place::Inside),
    }
}

/// Sort by an exact comparison that can fail. Two points that compare equal are one point.
fn order(
    indices: &mut [usize],
    compare: impl Fn(&usize, &usize) -> Result<Ordering, &'static str>,
) -> Result<(), &'static str> {
    for i in 1..indices.len() {
        let mut j = i;
        while j > 0 {
            match compare(&indices[j - 1], &indices[j])? {
                Ordering::Greater => indices.swap(j - 1, j),
                Ordering::Equal => return Err("two points on an edge are the same point"),
                Ordering::Less => break,
            }
            j -= 1;
        }
    }
    Ok(())
}

/// Positions along an edge, nearest to the true ones, that strictly increase from zero to one.
/// On the grid, they are multiples of its spacing.
fn increasing(approximate: &[f64], grid: bool) -> Result<Vec<f64>, &'static str> {
    let mut found: Vec<f64> = Vec::with_capacity(approximate.len());
    let mut previous = 0.0_f64;
    for &t in approximate {
        let wanted = if grid { (t * GRID).round() / GRID } else { t };
        let least = if grid {
            previous + 1.0 / GRID
        } else {
            f64::from_bits(previous.to_bits() + 1)
        };
        let t = wanted.max(least);
        if t >= 1.0 {
            return Err("the points on an edge cannot be told apart in the frame");
        }
        found.push(t);
        previous = t;
    }
    Ok(found)
}
