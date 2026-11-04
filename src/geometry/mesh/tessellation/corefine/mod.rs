#[cfg(test)]
mod test;

mod chain;
mod clip;
mod overlap;
mod pierce;

pub use chain::Polyline;
pub use overlap::{Constraint, Side};

use super::Tessellation;
use crate::geometry::{Coordinate, CoordinatesRef};
use chain::chain;
use pierce::{Local, Meeting, crossing};

type Triangle<'a> = [&'a Coordinate<3>; 3];

/// A vertex, edge or triangle of one tessellation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) enum Feature {
    Vertex(usize),
    Edge([usize; 2]),
    Face(usize),
}

/// The lowest-dimensional feature of each tessellation that a point lies on. Two ends with the
/// same key are the same point, however many triangles share it and however it was rounded.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct Key {
    first: Feature,
    second: Feature,
}

/// A segment along which a triangle of one tessellation crosses a triangle of another.
pub struct Crossing {
    pub first: usize,
    pub second: usize,
    pub segment: [Coordinate<3>; 2],
    pub(super) keys: [Key; 2],
}

/// How the triangles of one tessellation meet those of another.
struct Contacts {
    /// The segments along which triangles cross.
    crossings: Vec<Crossing>,
    /// The pairs of triangles, as indices into each tessellation, that lie in one plane.
    coplanar: Vec<[usize; 2]>,
    /// The segments that split triangles where the tessellations overlap in one plane.
    constraints: Vec<Constraint>,
}

impl Tessellation {
    /// The curves along which this tessellation intersects another.
    ///
    /// Where triangles of the two lie in one plane, the boundary of that overlap is included, but
    /// not what lies inside it; see [`Self::coplanar_pairs`].
    pub fn intersection_curves(&self, other: &Tessellation) -> Result<Vec<Polyline>, &'static str> {
        chain(&self.crossings(other)?)
    }
    /// Every segment along which this tessellation crosses another.
    ///
    /// Two triangles in one plane meet in an area, not a segment, so they have no crossing; the
    /// segments where their neighbours meet them are included. Fails on degenerate triangles.
    pub fn crossings(&self, other: &Tessellation) -> Result<Vec<Crossing>, &'static str> {
        Ok(self.contacts(other)?.crossings)
    }
    /// The pairs of triangles, as indices into this tessellation and the other, that lie in one
    /// plane and whose bounding boxes overlap. Some pairs only touch along an edge.
    pub fn coplanar_pairs(&self, other: &Tessellation) -> Result<Vec<[usize; 2]>, &'static str> {
        Ok(self.contacts(other)?.coplanar)
    }
    /// The segments that split the triangles of this tessellation and the other where they overlap
    /// in one plane, so that both are cut along each other's edges there.
    ///
    /// Each is a piece of an edge of one tessellation that crosses the inside of a coplanar
    /// triangle of the other, with its ends named by the features they lie on, as for
    /// [`Crossing`]. Pieces along an edge of the triangle they would split are not included.
    pub fn coplanar_constraints(
        &self,
        other: &Tessellation,
    ) -> Result<Vec<Constraint>, &'static str> {
        Ok(self.contacts(other)?.constraints)
    }
    fn contacts(&self, other: &Tessellation) -> Result<Contacts, &'static str> {
        let (coordinates, other_coordinates) =
            (self.mesh().coordinates(), other.mesh().coordinates());
        let triangles: Vec<&[usize]> = self.mesh().connectivities().iter().flatten().collect();
        let other_triangles: Vec<&[usize]> =
            other.mesh().connectivities().iter().flatten().collect();
        let other_bvh = other.bvh();
        let mut crossings = Vec::new();
        let mut coplanar = Vec::new();
        let mut constraints = Vec::new();
        for (first, nodes) in triangles.iter().enumerate() {
            let triangle = [
                &coordinates[nodes[0]],
                &coordinates[nodes[1]],
                &coordinates[nodes[2]],
            ];
            let bbox = triangle
                .into_iter()
                .collect::<CoordinatesRef<'_, 3>>()
                .into();
            for second in other_bvh.overlapping(&bbox) {
                let other_nodes = other_triangles[second];
                let other_triangle = [
                    &other_coordinates[other_nodes[0]],
                    &other_coordinates[other_nodes[1]],
                    &other_coordinates[other_nodes[2]],
                ];
                match crossing(triangle, other_triangle)? {
                    Meeting::Apart => (),
                    Meeting::Coplanar => {
                        coplanar.push([first, second]);
                        constraints.extend(overlap::constraints(
                            &overlap::Part {
                                triangle,
                                nodes,
                                index: first,
                            },
                            &overlap::Part {
                                triangle: other_triangle,
                                nodes: other_nodes,
                                index: second,
                            },
                        ));
                    }
                    Meeting::Segment(ends) => {
                        let keys = ends.each_ref().map(|end| Key {
                            first: feature(end.first, nodes, first),
                            second: feature(end.second, other_nodes, second),
                        });
                        crossings.push(Crossing {
                            first,
                            second,
                            keys,
                            segment: ends.map(|end| end.point),
                        });
                    }
                }
            }
        }
        Ok(Contacts {
            crossings,
            coplanar,
            constraints: overlap::unique(constraints),
        })
    }
}

fn feature(local: Local, nodes: &[usize], triangle: usize) -> Feature {
    match local {
        Local::Vertex(index) => Feature::Vertex(nodes[index]),
        Local::Edge(index) => {
            let (a, b) = (nodes[index], nodes[(index + 1) % 3]);
            Feature::Edge([a.min(b), a.max(b)])
        }
        Local::Face => Feature::Face(triangle),
    }
}
