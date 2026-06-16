#[cfg(test)]
mod test;

mod voxels;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh},
    },
    math::Set,
};
use std::cell::OnceCell;

impl<const D: usize> From<(Connectivities, Set<Coordinates<D>>)> for Mesh<D> {
    fn from((connectivities, coordinates): (Connectivities, Set<Coordinates<D>>)) -> Self {
        Self {
            connectivities,
            coordinates,
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
            nodes_elements: OnceCell::new(),
            nodes_nodes: OnceCell::new(),
        }
    }
}
