#[cfg(test)]
mod test;

use super::Tessellation;
use crate::{
    geometry::{
        Coordinate,
        solid::{Projection, SolidOracle},
    },
    math::{CrossProduct, Quantity},
    units::{Length, Volume},
};

/// What keeps a tessellation from bounding a volume that points can be inside or outside of.
#[derive(Debug)]
pub struct Defects {
    /// Edges of a single face, so the surface is open.
    pub boundary_edges: Vec<[usize; 2]>,
    /// Edges shared by more than two faces.
    pub non_manifold_edges: Vec<[usize; 2]>,
    /// Vertices where the faces around them form more than one fan.
    pub non_manifold_vertices: Vec<usize>,
    /// Directed edges traversed twice, where neighbouring faces disagree on orientation.
    pub inconsistent_edges: Vec<[usize; 2]>,
    /// Pairs of faces that cross without sharing a vertex.
    pub self_intersections: Vec<[usize; 2]>,
    /// Whether the faces enclose no positive volume, so they are oriented inward.
    pub inward: bool,
}

impl Defects {
    pub fn is_empty(&self) -> bool {
        self.boundary_edges.is_empty()
            && self.non_manifold_edges.is_empty()
            && self.non_manifold_vertices.is_empty()
            && self.inconsistent_edges.is_empty()
            && self.self_intersections.is_empty()
            && !self.inward
    }
}

impl Tessellation {
    /// Everything that keeps this tessellation from being closed, manifold, non-self-intersecting
    /// and outward oriented, which inside/outside queries by ray parity rely on.
    pub fn defects(&self) -> Defects {
        let mesh = self.mesh();
        Defects {
            boundary_edges: mesh.boundary_edges(),
            non_manifold_edges: mesh.non_manifold_edges(),
            non_manifold_vertices: mesh.non_manifold_vertices(),
            inconsistent_edges: mesh.inconsistent_edges(),
            self_intersections: mesh.self_intersections(),
            inward: self.enclosed_volume() <= Quantity::default(),
        }
    }
    /// The volume enclosed by the faces, positive when they are oriented outward.
    pub fn enclosed_volume(&self) -> Quantity<Volume> {
        let coordinates = self.mesh().coordinates();
        self.mesh()
            .connectivities()
            .iter()
            .flatten()
            .map(|face| {
                coordinates[face[1]].cross(&coordinates[face[2]]) * &coordinates[face[0]] / 6.0
            })
            .sum()
    }
}

/// A tessellation with no [`Defects`], so it bounds a volume.
pub struct Watertight(Tessellation);

impl Watertight {
    #[allow(clippy::result_large_err)]
    pub fn new(tessellation: Tessellation) -> Result<Self, Defects> {
        let defects = tessellation.defects();
        defects
            .is_empty()
            .then_some(Self(tessellation))
            .ok_or(defects)
    }
    pub fn tessellation(&self) -> &Tessellation {
        &self.0
    }
    pub fn into_inner(self) -> Tessellation {
        self.0
    }
}

impl SolidOracle for Watertight {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        self.0.signed_distance(point)
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        self.0.project(point)
    }
}
