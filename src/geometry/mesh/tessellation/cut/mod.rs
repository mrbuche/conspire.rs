mod assemble;
mod classify;
mod face;
mod geometry;
mod snap;
mod tables;

#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate,
        mesh::{
            Mesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::Scalar,
};
use geometry::contained;
use std::collections::HashMap;

const COLLAPSE_FRACTION: Scalar = 0.2;
const CROSSING_TOLERANCE: Scalar = 1.0e-8;
const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const PADDING: u16 = 2;
const SLIVER_FRACTION: Scalar = 0.1;
const SNAP_HARD: Scalar = 0.05;
const SNAP_QUALITY: Scalar = 0.3;
const SNAP_SOFT: Scalar = 0.2;
const FACES: [[usize; 4]; 6] = [
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [0, 3, 2, 1],
    [4, 5, 6, 7],
];
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];
const DIRECTIONS: [Coordinate<D>; 3] = [
    Coordinate::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Coordinate::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Coordinate::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Class {
    Inside,
    Cut,
    Outside,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sign {
    Inside,
    On,
    Outside,
}

/// Identifies a point used while stitching a cut face/cell.
///
/// Either an original mesh node, or the `ordinal`-th crossing,
/// (in canonical ascending-node-order direction) along the sorted edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Vertex {
    Node(usize),
    Crossing([usize; 2], usize),
}

pub struct Tables {
    signs: HashMap<usize, Sign>,
    crossings: HashMap<[usize; 2], Vec<Coordinate<D>>>,
    faces: HashMap<[usize; 4], [usize; 4]>,
    segments: HashMap<[usize; 4], Vec<[Vertex; 2]>>,
}

impl Tables {
    pub fn signs(&self) -> &HashMap<usize, Sign> {
        &self.signs
    }
    pub fn crossings(&self) -> &HashMap<[usize; 2], Vec<Coordinate<D>>> {
        &self.crossings
    }
    pub fn faces(&self) -> &HashMap<[usize; 4], [usize; 4]> {
        &self.faces
    }
    pub fn segments(&self) -> &HashMap<[usize; 4], Vec<[Vertex; 2]>> {
        &self.segments
    }
}

impl Tessellation {
    pub fn cut(&self, balancing: Balancing, scale: Scalar) -> Result<Mesh<D>, &'static str> {
        let mut octree =
            Octree::<u16, usize>::from_features(self, scale, CurvatureSizing::default(), PADDING);
        octree.equilibrate(balancing, Pairing::Regular)?;
        let mesh = octree.dualize();
        let classes = self.classify(&mesh);
        if !contained(&mesh, &classes) {
            return Err("tessellation is not contained within the dual mesh");
        }
        let (mesh, snapped) = self.snap(mesh, &classes)?;
        let tables = self.tables(&mesh, &classes, &snapped)?;
        self.assemble(&mesh, &classes, &tables)
    }
}
