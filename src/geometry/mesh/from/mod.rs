#[cfg(test)]
mod test;

mod lattice;
mod ntree;
mod pixels;
mod segmentation;
mod voxels;

pub(crate) use ntree::Dualization;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh, NodeSets, SideSets},
    },
    math::{CrossProduct, Set},
};
use std::cell::OnceCell;

/// The axis orders of the Kuhn/Freudenthal split: each a path of unit steps
/// from a cell's all-low corner to its all-high one.
const KUHN: [[usize; 2]; 6] = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]];

/// Splits a cell into six tetrahedra about the diagonal from corner `0` to
/// corner `7` of `corners`, indexed by the bits of each corner's position.
///
/// Bit `b` is set on the high side of axis `b`, so corners `0` and `7` are the
/// cell's lexicographic extremes and every square face ends up cut by the
/// diagonal joining its own two extremes. That is a property of the face, not
/// of either cell holding it, so neighbors cut a shared face identically.
pub(super) fn kuhn(corners: &[usize; 8]) -> [[usize; 4]; 6] {
    KUHN.map(|[first, second]| {
        [
            corners[0],
            corners[1 << first],
            corners[(1 << first) | (1 << second)],
            corners[7],
        ]
    })
}

pub(super) fn positive(tet: &[usize; 4], coordinates: &Coordinates<3>) -> bool {
    let u = &coordinates[tet[1]] - &coordinates[tet[0]];
    let v = &coordinates[tet[2]] - &coordinates[tet[0]];
    let w = &coordinates[tet[3]] - &coordinates[tet[0]];
    (&u.cross(v) * &w).value() > 0.0
}

pub(super) fn orient(tets: &mut [[usize; 4]], coordinates: &Coordinates<3>) {
    tets.iter_mut().for_each(|tet| {
        if !positive(tet, coordinates) {
            tet.swap(2, 3)
        }
    })
}

impl<const D: usize> From<(Connectivities, Set<Coordinates<D>>)> for Mesh<D> {
    fn from((connectivities, coordinates): (Connectivities, Set<Coordinates<D>>)) -> Self {
        Self {
            connectivities,
            coordinates,
            node_sets: NodeSets::from(Vec::new()),
            side_sets: SideSets::from(Vec::new()),
            nodes_elements: OnceCell::new(),
            nodes_nodes: OnceCell::new(),
        }
    }
}

impl<const D: usize> From<(Vec<Connectivity>, Coordinates<D>)> for Mesh<D> {
    fn from((connectivities, coordinates): (Vec<Connectivity>, Coordinates<D>)) -> Self {
        Self {
            connectivities: Connectivities::from(connectivities),
            coordinates: Set::from(coordinates),
            node_sets: NodeSets::from(Vec::new()),
            side_sets: SideSets::from(Vec::new()),
            nodes_elements: OnceCell::new(),
            nodes_nodes: OnceCell::new(),
        }
    }
}
