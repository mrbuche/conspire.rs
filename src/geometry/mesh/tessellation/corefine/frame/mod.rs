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

use super::{Feature, triangulate::Input};
use crate::{
    geometry::{
        Coordinate,
        predicates::{Cutter, along, orient2d_planar},
    },
    math::Quantity,
};
use alloc::collections::{BTreeMap, BTreeSet};
use core::cmp::Ordering;

/// Edge points are placed at multiples of this on the edge from `(1, 0)` to `(0, 1)`, so that
/// both of their coordinates, which sum to one, are exact.
const GRID: f64 = (1u64 << 52) as f64;

/// How many times a point rounded onto or past the boundary is nudged inwards, by a fraction of
/// its distance to the middle that goes from 2⁻⁵² to 2⁻³² of it.
const NUDGES: usize = 11;

/// A triangle in its frame, with the name of every point that is not a corner.
pub(super) struct Frame<K> {
    pub input: Input,
    /// The key of each point after the three corners, in the order of the points.
    pub keys: Vec<K>,
}

enum Place {
    Corner(usize),
    Edge(usize),
    Inside,
}

/// The frame of a triangle with corners `triangle`, named by `nodes`, for the points that lie on it
/// and the segments between them.
///
/// Every point has a key. Its `feature` is the lowest-dimension feature of the triangle's
/// tessellation that it lies on, which says where it is on the triangle, and its `cutters` are
/// what cut an edge at it, any one of which is enough to order the points of an edge, the first
/// that is not parallel to the edge being used. A point on a corner need not be listed. The ranks
/// of the points are the order of their keys.
pub(super) fn frame<'a, K: Ord + Clone>(
    triangle: [&Coordinate<3>; 3],
    nodes: [usize; 3],
    points: &[(K, Coordinate<3>)],
    segments: &[[K; 2]],
    feature: impl Fn(&K) -> &Feature,
    cutters: impl Fn(&K) -> Vec<Cutter<'a>>,
) -> Result<Frame<K>, &'static str> {
    let first = triangle[1] - triangle[0];
    let second = triangle[2] - triangle[0];
    let gram = [&first * &first, &first * &second, &second * &second];
    let determinant = gram[0] * gram[2] - gram[1] * gram[1];
    if determinant.partial_cmp(&Quantity::default()) != Some(Ordering::Greater)
        || !determinant.value().is_finite()
    {
        return Err("the triangle is degenerate");
    }
    let place = |key: &K| place(feature(key), &nodes);
    let mut found = BTreeMap::<&K, &Coordinate<3>>::new();
    for (key, position) in points {
        if !matches!(place(key)?, Place::Corner(_)) {
            found.insert(key, position);
        }
    }
    let keys: Vec<K> = found.keys().map(|&key| key.clone()).collect();
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
            let position = found[key] - triangle[0];
            let (a, b) = (&first * &position, &second * &position);
            let point = [
                ((a * gram[2] - b * gram[1]) / determinant).value(),
                ((b * gram[0] - a * gram[1]) / determinant).value(),
            ];
            frame_points[index + 3] = within(point)?;
        }
    }
    for (edge, indices) in sides.iter_mut().enumerate() {
        let (p, q) = (triangle[edge], triangle[(edge + 1) % 3]);
        order(indices, |&a, &b| {
            cut_order(p, q, &cutters(&keys[a - 3]), &cutters(&keys[b - 3]))
        })?;
        let length = q - p;
        let approximate: Vec<f64> = indices
            .iter()
            .map(|&index| {
                let from_start = found[&keys[index - 3]] - p;
                ((&from_start * &length) / (&length * &length)).value()
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
    let locate = |key: &K| -> Result<usize, &'static str> {
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
            gram: gram.map(|g| g.value()),
            points: frame_points,
            sides,
            segments,
        },
        keys,
    })
}

/// Which of two points on an edge from `p` to `q` is first, from what cuts the edge at each.
pub(super) fn cut_order(
    p: &Coordinate<3>,
    q: &Coordinate<3>,
    first: &[Cutter],
    second: &[Cutter],
) -> Result<Ordering, &'static str> {
    let mut found = Err("a point on an edge has nothing to cut the edge");
    for (x, y) in first
        .iter()
        .flat_map(|x| second.iter().map(move |y| (x, y)))
    {
        found = along(p, q, x, y);
        if found.is_ok() {
            break;
        }
    }
    found
}

/// The pairs of the points given, with their keys, that are the same point, because what cuts the
/// edge they are on cuts it at the same place. The points are placed in the same frame, in which
/// two such points cannot be told apart.
pub(super) fn coincident<'a, K: Ord + Clone>(
    triangle: [&Coordinate<3>; 3],
    nodes: [usize; 3],
    points: &[(K, Coordinate<3>)],
    feature: impl Fn(&K) -> &Feature,
    cutters: impl Fn(&K) -> Vec<Cutter<'a>>,
) -> Result<Vec<[K; 2]>, &'static str> {
    let mut sides: [Vec<&K>; 3] = Default::default();
    let keys: BTreeSet<&K> = points.iter().map(|(key, _)| key).collect();
    for key in keys {
        if let Place::Edge(edge) = place(feature(key), &nodes)? {
            sides[edge].push(key);
        }
    }
    let mut same = Vec::new();
    for (edge, keys) in sides.iter_mut().enumerate() {
        let (p, q) = (triangle[edge], triangle[(edge + 1) % 3]);
        for i in 1..keys.len() {
            let mut j = i;
            while j > 0 {
                match cut_order(p, q, &cutters(keys[j - 1]), &cutters(keys[j]))? {
                    Ordering::Greater => keys.swap(j - 1, j),
                    Ordering::Equal => {
                        same.push([keys[j - 1].clone(), keys[j].clone()]);
                        break;
                    }
                    Ordering::Less => break,
                }
                j -= 1;
            }
        }
    }
    Ok(same)
}

/// A point strictly inside the frame, nudged towards the middle if rounding left it on the
/// boundary or just outside. How far it may be nudged is far more than rounding, far less than
/// the size of the triangle. Beyond that the point is not where it should be, and it fails.
fn within(point: [f64; 2]) -> Result<[f64; 2], &'static str> {
    let middle = [1.0 / 3.0; 2];
    let strictly = |p: [f64; 2]| {
        [
            orient2d_planar([0.0, 0.0], [1.0, 0.0], p),
            orient2d_planar([1.0, 0.0], [0.0, 1.0], p),
            orient2d_planar([0.0, 1.0], [0.0, 0.0], p),
        ]
        .iter()
        .all(|&sign| sign == Ordering::Greater)
    };
    (0..=NUDGES)
        .map(|step| {
            let fraction = if step == 0 {
                0.0
            } else {
                f64::powi(2.0, -(52 - 2 * (step as i32 - 1)))
            };
            [0, 1].map(|i| point[i] + (middle[i] - point[i]) * fraction)
        })
        .find(|&candidate| strictly(candidate))
        .ok_or("a point inside the triangle is rounded out of it")
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
