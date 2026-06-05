#[cfg(test)]
pub mod test;

mod base;
mod connectivity;
mod from;
mod into;
mod read;
mod tessellation;
mod write;

pub use self::{
    connectivity::{Connectivities, Connectivity},
    read::Input,
    tessellation::Tessellation,
    write::Output,
};

use crate::{geometry::Coordinates, math::Set};
use std::cell::OnceCell;

// need to "merge" node-to-element connectivity since node in block 1 can touch element in block 2
// then would be able to get the node-to-node connectivity
// would only want to compute that once and store, similar to Sets
// could this be involved in a Graph type?

pub struct Mesh<const D: usize> {
    connectivities: Connectivities,
    coordinates: Set<Coordinates<D>>,
    nodes_elements: OnceCell<Vec<Vec<usize>>>,
    nodes_nodes: OnceCell<Vec<Vec<usize>>>,
}
