#[cfg(test)]
pub(crate) mod test;

pub mod cohesive;
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
    domain::block::{add_node_neighbors, element_coordinates, trimmed_type_name},
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
    R = NoDensity,
    H = NoHeatCapacity,
> {
    constitutive_model: C,
    connectivity: PrimitiveConnectivity<M, N>,
    elements: Vec<F>,
    density: R,
    heat_capacity: H,
}

impl<C, F, const G: usize, const M: usize, const N: usize, R, H> Block<C, F, G, M, N, R, H>
where
    F: FiniteElement<G, M, N>,
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
        element_coordinates(coordinates, nodes)
    }
    pub fn volume(&self) -> Quantity<Volume> {
        self.elements().iter().map(|element| element.volume()).sum()
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, R, H> Debug
    for Block<C, F, G, M, N, R, H>
where
    F: FiniteElement<G, M, N>,
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

impl<C, F, const G: usize, const M: usize, const N: usize, R, H> Elements
    for Block<C, F, G, M, N, R, H>
where
    F: FiniteElement<G, M, N>,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(
            self.connectivity().iter().map(|nodes| nodes.as_slice()),
            neighbors,
        )
    }
}

impl<C, F, const G: usize, const N: usize>
    From<(
        C,
        PrimitiveConnectivity<3, N>,
        &NodalReferenceCoordinates<3>,
    )> for Block<C, F, G, 3, N>
where
    F: FiniteElement<G, 3, N> + From<ElementNodalReferenceCoordinates<N>>,
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
        }
    }
}

impl<C, F, const G: usize, const N: usize> From<(C, Vec<[usize; N]>, &NodalReferenceCoordinates<3>)>
    for Block<C, F, G, 3, N>
where
    F: FiniteElement<G, 3, N> + From<ElementNodalReferenceCoordinates<N>>,
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

impl<C, F, const G: usize, const N: usize>
    From<(
        C,
        PrimitiveConnectivity<2, N>,
        &NodalReferenceCoordinates<2>,
    )> for Block<C, F, G, 2, N>
where
    F: FiniteElement<G, 2, N> + From<PlanarElementNodalReferenceCoordinates<N>>,
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
        }
    }
}

impl<C, F, const G: usize, const N: usize> From<(C, Vec<[usize; N]>, &NodalReferenceCoordinates<2>)>
    for Block<C, F, G, 2, N>
where
    F: FiniteElement<G, 2, N> + From<PlanarElementNodalReferenceCoordinates<N>>,
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
