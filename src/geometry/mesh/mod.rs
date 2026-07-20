#[cfg(test)]
pub mod test;

mod base;
mod connectivity;
mod differential;
mod from;
mod into;
mod quality;
mod read;
mod remesh;
mod smooth;
mod tessellation;
mod write;

pub use self::{
    connectivity::{
        Connectivities, Connectivity, polytopal::PolytopalConnectivity,
        primitive::PrimitiveConnectivity,
    },
    differential::laplace::Weighting,
    quality::metrics::Verdict,
    read::Input,
    remesh::{AnisotropicSizing, IsotropicSizing, Remeshing, RemeshingMetric},
    smooth::Smoothing,
    tessellation::Tessellation,
    write::{
        Output,
        vtk::{UnstructuredGrid, Vtk},
    },
};

use crate::{
    geometry::Coordinates,
    math::{Graph, Set},
};
use std::cell::OnceCell;

pub type NodeSets = Set<Vec<Vec<usize>>>;
pub type SideSets = Set<Vec<Vec<(usize, usize)>>>;

pub struct Mesh<const D: usize> {
    connectivities: Connectivities,
    coordinates: Set<Coordinates<D>>,
    node_sets: NodeSets,
    side_sets: SideSets,
    nodes_elements: OnceCell<Vec<Vec<usize>>>,
    nodes_nodes: OnceCell<Graph>,
}
