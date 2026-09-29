#[cfg(test)]
pub mod test;

mod base;
mod buffer;
mod connectivity;
pub(crate) mod differential;
mod fit;
mod from;
mod into;
mod partition;
mod patch;
mod quality;
mod read;
mod remesh;
mod retain;
mod sampling;
mod sets;
mod smooth;
mod tessellation;
mod write;

pub use from::Dualization;
pub use sets::{NodeSet, NodeSets, SideSet, SideSets};

pub use self::{
    connectivity::{
        Connectivities, Connectivity, polytopal::PolytopalConnectivity,
        primitive::PrimitiveConnectivity,
    },
    differential::laplace::Weighting,
    fit::{Fitting, Freedom},
    partition::{Bisection, Partition, PartitionQuality},
    patch::Patch,
    quality::metrics::Verdict,
    read::Input,
    remesh::{AnisotropicSizing, IsotropicSizing, Remeshing, RemeshingMetric},
    smooth::Smoothing,
    tessellation::{
        Tessellation,
        cut::{
            Class,
            marching::{Finish, Marching, Placement},
        },
        write::Stl,
    },
    write::{Output, exodus::ExodusFormat, vtk::Vtk},
};

use crate::{
    geometry::Coordinates,
    math::{Incidence, Labels},
};
use std::cell::OnceCell;

pub struct Mesh<const D: usize> {
    connectivities: Connectivities,
    coordinates: Coordinates<D>,
    node_labels: Option<Labels>,
    node_sets: NodeSets,
    side_sets: SideSets,
    nodes_elements: OnceCell<Incidence>,
    nodes_nodes: OnceCell<Incidence>,
}
