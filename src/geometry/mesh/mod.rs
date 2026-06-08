#[cfg(test)]
pub mod test;

mod base;
mod connectivity;
mod differential;
mod from;
mod into;
mod read;
mod remesh;
mod smooth;
mod tessellation;
mod write;

pub use self::{
    connectivity::{Connectivities, Connectivity},
    read::Input,
    tessellation::Tessellation,
    write::Output,
};

use crate::{
    geometry::Coordinates,
    math::{Graph, Set},
};
use std::cell::OnceCell;

pub struct Mesh<const D: usize> {
    connectivities: Connectivities,
    coordinates: Set<Coordinates<D>>,
    nodes_elements: OnceCell<Vec<Vec<usize>>>,
    nodes_nodes: OnceCell<Graph>,
}
