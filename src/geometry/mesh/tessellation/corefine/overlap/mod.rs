#[cfg(test)]
mod test;

use super::{
    Key, Triangle,
    clip::{Cut, clip},
    feature,
    pierce::Local,
};
use crate::{
    geometry::Coordinate,
    math::{CrossProduct, FxHashSet},
};

/// Which of the two tessellations a triangle belongs to.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Side {
    First,
    Second,
}

/// A segment that splits a triangle where an edge of the other tessellation crosses it, within a
/// patch where the two tessellations lie in one plane.
pub struct Constraint {
    /// The tessellation the split triangle belongs to.
    pub side: Side,
    /// The index of the split triangle in its tessellation.
    pub triangle: usize,
    /// The two ends of the segment, rounded: only the features they lie on are exact.
    pub segment: [Coordinate<3>; 2],
    pub(super) keys: [Key; 2],
}

/// A triangle of one tessellation and the vertices that name its features.
pub(super) struct Part<'a> {
    pub triangle: Triangle<'a>,
    pub nodes: &'a [usize],
    pub index: usize,
}

/// The pieces of the edges of each of two coplanar triangles that lie inside the other, where
/// they cross its inside rather than run along its own edges.
pub(super) fn constraints(first: &Part, second: &Part) -> Vec<Constraint> {
    let axis = dominant_axis(first.triangle);
    let mut found = pieces(second, first, Side::First, axis);
    found.extend(pieces(first, second, Side::Second, axis));
    found
}

/// The pieces of the edges of `cutter` that cross the inside of `cut`, as constraints on `cut`.
fn pieces(cutter: &Part, cut: &Part, side: Side, axis: usize) -> Vec<Constraint> {
    let mut found = Vec::new();
    for edge in 0..3 {
        let next = (edge + 1) % 3;
        let Some([start, finish]) = clip(
            cutter.triangle[edge],
            cutter.triangle[next],
            cut.triangle,
            axis,
        ) else {
            continue;
        };
        if shares_an_edge(start.triangle, finish.triangle) {
            continue;
        }
        let key = |end: &Cut| {
            let on_cutter = match end.segment {
                Local::Vertex(0) => Local::Vertex(edge),
                Local::Vertex(_) => Local::Vertex(next),
                _ => Local::Edge(edge),
            };
            let on_cut = feature(end.triangle, cut.nodes, cut.index);
            let on_cutter = feature(on_cutter, cutter.nodes, cutter.index);
            match side {
                Side::First => Key {
                    first: on_cut,
                    second: on_cutter,
                },
                Side::Second => Key {
                    first: on_cutter,
                    second: on_cut,
                },
            }
        };
        found.push(Constraint {
            side,
            triangle: cut.index,
            keys: [key(&start), key(&finish)],
            segment: [start.point, finish.point],
        });
    }
    found
}

/// Whether two points of a triangle lie on one of its edges, so the segment between them runs
/// along the boundary of the triangle instead of across its inside.
fn shares_an_edge(a: Local, b: Local) -> bool {
    let edges = |local: Local| match local {
        Local::Face => Vec::new(),
        Local::Edge(edge) => vec![edge],
        Local::Vertex(vertex) => vec![vertex, (vertex + 2) % 3],
    };
    let (a, b) = (edges(a), edges(b));
    a.iter().any(|edge| b.contains(edge))
}

/// A coordinate along which the normal of the plane of the triangle is largest.
fn dominant_axis(triangle: Triangle) -> usize {
    let normal = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
    (0..3)
        .max_by(|&a, &b| normal[a].abs().total_cmp(&normal[b].abs()))
        .unwrap_or(2)
}

/// The constraints with each segment kept once, however many pairs of triangles found it.
pub(super) fn unique(constraints: Vec<Constraint>) -> Vec<Constraint> {
    let mut seen = FxHashSet::<(Side, usize, Key, Key)>::default();
    constraints
        .into_iter()
        .filter(|constraint| {
            let [a, b] = constraint.keys.clone();
            !seen.contains(&(constraint.side, constraint.triangle, b.clone(), a.clone()))
                && seen.insert((constraint.side, constraint.triangle, a, b))
        })
        .collect()
}
