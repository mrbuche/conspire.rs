//! Boundary-representation solid models.
//!
//! Topology (shells of faces, faces bounded by edge loops, edges between
//! vertices) layered over geometry (`Surface`, `Curve`, points). The STEP
//! reader builds one of these; meshing consumes it.

#[cfg(test)]
pub mod test;

pub mod curve;
pub mod features;
pub mod inside;
pub mod planar;
pub mod surface;
mod tessellate;

use crate::geometry::Coordinate;
use curve::Curve;
use std::array::from_fn;
use surface::Surface;

const D: usize = 3;

pub struct Brep {
    pub vertices: Vec<Coordinate<D>>,
    pub edges: Vec<Edge>,
    pub faces: Vec<Face>,
    pub shells: Vec<Shell>,
}

pub struct Edge {
    pub vertices: [usize; 2],
    pub curve: Curve,
}

pub struct HalfEdge {
    pub edge: usize,
    pub forward: bool,
}

pub struct Loop {
    pub half_edges: Vec<HalfEdge>,
}

pub struct Face {
    pub surface: Surface,
    /// Outer loop first, then inner (hole) loops.
    pub bounds: Vec<Loop>,
    /// Whether the face normal agrees with its surface normal.
    pub forward: bool,
}

pub struct Shell {
    pub faces: Vec<usize>,
    pub closed: bool,
}

impl Face {
    /// The outward unit normal. Planar faces only.
    fn normal(&self) -> [f64; D] {
        let Surface::Plane(plane) = &self.surface;
        let sign = if self.forward { 1.0 } else { -1.0 };
        from_fn(|axis| sign * plane.normal[axis].value())
    }
}

impl Loop {
    /// The loop's vertices in traversal order, one per half-edge.
    fn vertices(&self, edges: &[Edge]) -> Result<Vec<usize>, &'static str> {
        self.half_edges
            .iter()
            .map(|half_edge| {
                let edge = edges
                    .get(half_edge.edge)
                    .ok_or("half-edge references a missing edge")?;
                Ok(edge.vertices[usize::from(!half_edge.forward)])
            })
            .collect()
    }
}
