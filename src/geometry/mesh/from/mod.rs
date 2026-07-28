#[cfg(test)]
mod test;

mod ntree;
mod pixels;
mod segmentation;
mod voxels;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh, NodeSets, SideSets},
    },
    math::Set,
};
use std::cell::OnceCell;

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
