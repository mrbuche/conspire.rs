#[cfg(test)]
pub(crate) mod test;

mod density;
pub(crate) mod diffusion;
pub mod electrical;
pub mod element;
pub mod mass;
pub mod multiphysics;
pub mod solid;
pub mod surface;
pub mod thermal;
mod time_scale;
pub mod volume;

use crate::{
    domain::block::{add_node_neighbors, trimmed_type_name},
    fem::{
        Elements, NodalReferenceCoordinates,
        block::element::{
            ElementNodalReferenceCoordinates, FiniteElement,
            planar::PlanarElementNodalReferenceCoordinates,
        },
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{Quantity, TensorRank1List, TensorRank1Vec},
    units::Volume,
};
use std::fmt::{self, Debug, Formatter};

pub use density::{Densities, DensityField, ElementDensities, NoDensity};
pub use thermal::capacity::{ElementHeatCapacities, HeatCapacities, NoHeatCapacity};

pub struct Block<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
    R = NoDensity,
    H = NoHeatCapacity,
> {
    constitutive_model: C,
    connectivity: PrimitiveConnectivity<M, N>,
    elements: Vec<F>,
    density: R,
    heat_capacity: H,
    scratch: Scratch<N>,
}

/// TEMPORARY EXPERIMENT: element forces kept between calls so the parallel pass does not
/// allocate, and the nodes' incident (element, local node) pairs for a deterministic gather.
pub(crate) struct Scratch<const N: usize> {
    pub(crate) forces: std::sync::RwLock<Vec<crate::mechanics::ForceList<N>>>,
    pub(crate) gather: std::sync::OnceLock<Gather>,
}

/// For each node, the flat indices `N * element + local` of its contributions, in element order.
pub(crate) struct Gather {
    pub(crate) offsets: Vec<usize>,
    pub(crate) entries: Vec<usize>,
}

impl Gather {
    pub(crate) fn new<const N: usize>(connectivity: &[[usize; N]]) -> Self {
        let nodes = connectivity
            .iter()
            .flatten()
            .max()
            .map_or(0, |&node| node + 1);
        let mut offsets = vec![0; nodes + 1];
        connectivity
            .iter()
            .flatten()
            .for_each(|&node| offsets[node + 1] += 1);
        (0..nodes).for_each(|node| offsets[node + 1] += offsets[node]);
        let mut next = offsets.clone();
        let mut entries = vec![0; connectivity.len() * N];
        connectivity
            .iter()
            .enumerate()
            .for_each(|(element, nodes)| {
                nodes.iter().enumerate().for_each(|(local, &node)| {
                    entries[next[node]] = N * element + local;
                    next[node] += 1;
                })
            });
        Self { offsets, entries }
    }
}

impl<const N: usize> Default for Scratch<N> {
    fn default() -> Self {
        Self {
            forces: std::sync::RwLock::new(Vec::new()),
            gather: std::sync::OnceLock::new(),
        }
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H>
    Block<C, F, G, M, N, P, R, H>
where
    F: FiniteElement<G, M, N, P>,
{
    pub(crate) fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    pub(crate) fn connectivity(&self) -> &PrimitiveConnectivity<M, N> {
        &self.connectivity
    }
    pub(crate) fn elements(&self) -> &[F] {
        &self.elements
    }
    pub(crate) fn element_coordinates<const D: usize, I, U>(
        coordinates: &TensorRank1Vec<D, I, U>,
        nodes: &[usize; N],
    ) -> TensorRank1List<D, I, N, U> {
        nodes
            .iter()
            .map(|&node| coordinates[node].clone())
            .collect()
    }
    pub fn volume(&self) -> Quantity<Volume> {
        self.elements().iter().map(|element| element.volume()).sum()
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H> Debug
    for Block<C, F, G, M, N, P, R, H>
where
    F: FiniteElement<G, M, N, P>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Block {{ constitutive model: {}, {} elements }}",
            trimmed_type_name::<C>(),
            self.elements().len()
        )
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H> Elements
    for Block<C, F, G, M, N, P, R, H>
where
    F: FiniteElement<G, M, N, P>,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(
            self.connectivity().iter().map(|nodes| nodes.as_slice()),
            neighbors,
        )
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(
        C,
        PrimitiveConnectivity<3, N>,
        &NodalReferenceCoordinates<3>,
    )> for Block<C, F, G, 3, N, P>
where
    F: FiniteElement<G, 3, N, P> + From<ElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            PrimitiveConnectivity<3, N>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|nodes| Self::element_coordinates(coordinates, nodes).into())
            .collect();
        Self {
            constitutive_model,
            connectivity,
            elements,
            density: NoDensity,
            heat_capacity: NoHeatCapacity,
            scratch: Scratch::default(),
        }
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(C, Vec<[usize; N]>, &NodalReferenceCoordinates<3>)> for Block<C, F, G, 3, N, P>
where
    F: FiniteElement<G, 3, N, P> + From<ElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            Vec<[usize; N]>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        (
            constitutive_model,
            PrimitiveConnectivity::from(connectivity),
            coordinates,
        )
            .into()
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(
        C,
        PrimitiveConnectivity<2, N>,
        &NodalReferenceCoordinates<2>,
    )> for Block<C, F, G, 2, N, P>
where
    F: FiniteElement<G, 2, N, P> + From<PlanarElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            PrimitiveConnectivity<2, N>,
            &NodalReferenceCoordinates<2>,
        ),
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|nodes| Self::element_coordinates(coordinates, nodes).into())
            .collect();
        Self {
            constitutive_model,
            connectivity,
            elements,
            density: NoDensity,
            heat_capacity: NoHeatCapacity,
            scratch: Scratch::default(),
        }
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(C, Vec<[usize; N]>, &NodalReferenceCoordinates<2>)> for Block<C, F, G, 2, N, P>
where
    F: FiniteElement<G, 2, N, P> + From<PlanarElementNodalReferenceCoordinates<N>>,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            Vec<[usize; N]>,
            &NodalReferenceCoordinates<2>,
        ),
    ) -> Self {
        (
            constitutive_model,
            PrimitiveConnectivity::from(connectivity),
            coordinates,
        )
            .into()
    }
}

pub(crate) use crate::domain::block::{finalize_node_neighbors, solver_from_neighbors};
