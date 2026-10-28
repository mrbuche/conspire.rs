#[cfg(test)]
mod test;

mod chain;
mod pierce;

pub use chain::Polyline;

use super::Tessellation;
use crate::geometry::{Coordinate, CoordinatesRef};
use chain::chain;
use pierce::{Local, crossing};

type Triangle<'a> = [&'a Coordinate<3>; 3];

/// A vertex, edge or triangle of one tessellation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) enum Feature {
    Vertex(usize),
    Edge([usize; 2]),
    Face(usize),
}

/// The lowest-dimensional feature of each tessellation that a point lies on. Two ends with the
/// same key are the same point, however many triangles share it and however it was rounded.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
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

impl Tessellation {
    /// The curves along which this tessellation intersects another.
    pub fn intersection_curves(&self, other: &Tessellation) -> Result<Vec<Polyline>, &'static str> {
        chain(&self.crossings(other)?)
    }
    /// Every segment along which this tessellation crosses another.
    ///
    /// Fails on coplanar or degenerate triangle pairs, which have no crossing segment.
    pub fn crossings(&self, other: &Tessellation) -> Result<Vec<Crossing>, &'static str> {
        let (coordinates, other_coordinates) =
            (self.mesh().coordinates(), other.mesh().coordinates());
        let triangles: Vec<&[usize]> = self.mesh().connectivities().iter().flatten().collect();
        let other_triangles: Vec<&[usize]> =
            other.mesh().connectivities().iter().flatten().collect();
        let other_bvh = other.bvh();
        let mut crossings = Vec::new();
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
                if let Some(ends) = crossing(triangle, other_triangle)? {
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
        Ok(crossings)
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
