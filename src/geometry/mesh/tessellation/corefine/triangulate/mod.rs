//! The constrained Delaunay triangulation of the points on and inside one triangle.
//!
//! The boundary of the triangle is kept: the points on each of its edges split it into sub-edges
//! that are never flipped. So are the segments asked for, which are recovered by flipping the
//! edges that cross them[^1] and then fixed. Every decision is exact, in the frame the points are
//! given in, with circles measured by a quadratic form. Where points are cocircular the choice is
//! made as if each point's height on the paraboloid were raised by a power of a tiny number that
//! falls with its rank, its index, so that the result is unique and does not depend on the order
//! of insertion.
//!
//! [^1]: S. W. Sloan, [Comput. Struct. **47**, 441 (1993)](https://doi.org/10.1016/0045-7949(93)90239-A).

mod delaunay;
mod insert;
mod recover;
#[cfg(test)]
mod test;

use crate::{geometry::predicates::orient2d_planar, math::FxHashSet};
use core::cmp::Ordering;

/// The points to triangulate, and the segments between them that must be edges.
#[derive(Clone)]
pub(super) struct Input {
    /// The quadratic form `[g₀, g₁, g₂]` that circles are measured with, as for `incircle_metric`.
    pub gram: [f64; 3],
    /// The points, with the three corners of the triangle first, counterclockwise. The index of a
    /// point is its rank, which decides cocircular ties.
    pub points: Vec<[f64; 2]>,
    /// The points strictly inside each edge of the triangle, from the corner the edge starts at,
    /// in order. Edge `i` runs from corner `i` to corner `i + 1`.
    pub sides: [Vec<usize>; 3],
    /// Pairs of points to join. A segment through other points is joined through them, and none
    /// may cross another in a point that is not one of the points.
    pub segments: Vec<[usize; 2]>,
}

impl Input {
    /// The points that are neither corners nor on an edge.
    fn interior(&self) -> Vec<usize> {
        let mut on_side = vec![false; self.points.len()];
        self.sides
            .iter()
            .flatten()
            .for_each(|&point| on_side[point] = true);
        (3..self.points.len())
            .filter(|&point| !on_side[point])
            .collect()
    }
}

/// The triangles, counterclockwise, each starting at its smallest index, in order.
#[cfg(test)]
pub(super) fn triangulate(input: &Input) -> Result<Vec<[usize; 3]>, &'static str> {
    Ok(build(input, [0, 1, 2], &input.interior())?.triangles())
}

type Constrained = (Vec<[usize; 3]>, Vec<[usize; 2]>);

/// The triangles as for `triangulate`, and the edges that the segments became, each with its
/// smaller index first, in order.
pub(super) fn triangulate_constrained(input: &Input) -> Result<Constrained, &'static str> {
    let triangulation = build(input, [0, 1, 2], &input.interior())?;
    let mut constrained: Vec<[usize; 2]> = triangulation.fixed.iter().copied().collect();
    constrained.sort_unstable();
    Ok((triangulation.triangles(), constrained))
}

/// The triangulation, inserting the points of the edges in the given order of edges, then the
/// points inside in the given order. The result does not depend on either.
fn build<'a>(
    input: &'a Input,
    edges: [usize; 3],
    interior: &[usize],
) -> Result<Triangulation<'a>, &'static str> {
    let mut triangulation = Triangulation::new(input)?;
    for edge in edges {
        for &point in &input.sides[edge] {
            triangulation.insert_on_side(edge, point)?;
        }
    }
    for &point in interior {
        triangulation.insert(point)?;
    }
    triangulation.recover(&input.segments)?;
    triangulation.legalize_all()?;
    Ok(triangulation)
}

/// What is across an edge of a triangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Across {
    /// Another triangle.
    Face(usize),
    /// The outside, across the edge of the original triangle with this index.
    Side(usize),
}

/// A triangle, counterclockwise, with the neighbour opposite each of its vertices.
#[derive(Clone, Copy)]
struct Face {
    nodes: [usize; 3],
    across: [Across; 3],
}

struct Triangulation<'a> {
    input: &'a Input,
    faces: Vec<Face>,
    /// The edges that may not flip, as pairs of points in increasing order.
    fixed: FxHashSet<[usize; 2]>,
}

impl<'a> Triangulation<'a> {
    fn new(input: &'a Input) -> Result<Self, &'static str> {
        if input.points.len() < 3 {
            return Err("a triangle needs three corners");
        }
        let corner = |index: usize| input.points[index];
        if orient2d_planar(corner(0), corner(1), corner(2)) != Ordering::Greater {
            return Err("the corners must run counterclockwise");
        }
        for (edge, points) in input.sides.iter().enumerate() {
            for &point in points {
                if point < 3
                    || orient2d_planar(corner(edge), corner((edge + 1) % 3), input.points[point])
                        != Ordering::Equal
                {
                    return Err("a point of an edge is not on the edge");
                }
            }
        }
        Ok(Self {
            input,
            faces: vec![Face {
                nodes: [0, 1, 2],
                across: [Across::Side(1), Across::Side(2), Across::Side(0)],
            }],
            fixed: FxHashSet::default(),
        })
    }
    fn point(&self, node: usize) -> [f64; 2] {
        self.input.points[node]
    }
    fn orient(&self, a: usize, b: usize, c: usize) -> Ordering {
        orient2d_planar(self.point(a), self.point(b), self.point(c))
    }
    /// The triangles, each starting at its smallest index, in order.
    fn triangles(&self) -> Vec<[usize; 3]> {
        let mut triangles: Vec<[usize; 3]> = self
            .faces
            .iter()
            .map(|face| {
                let start = (0..3).min_by_key(|&k| face.nodes[k]).unwrap_or(0);
                [0, 1, 2].map(|k| face.nodes[(start + k) % 3])
            })
            .collect();
        triangles.sort_unstable();
        triangles
    }
}
