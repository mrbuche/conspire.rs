#[cfg(test)]
mod test;

pub mod element;
pub mod solid;
pub mod surface;
pub mod thermal;

use crate::{
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
use std::{
    any::type_name,
    fmt::{self, Debug, Formatter},
};

pub struct Block<C, F, const G: usize, const M: usize, const N: usize, const P: usize> {
    constitutive_model: C,
    connectivity: PrimitiveConnectivity<M, N>,
    elements: Vec<F>,
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> Block<C, F, G, M, N, P>
where
    F: FiniteElement<G, M, N, P>,
{
    fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn connectivity(&self) -> &PrimitiveConnectivity<M, N> {
        &self.connectivity
    }
    fn elements(&self) -> &[F] {
        &self.elements
    }
    fn element_coordinates<const D: usize, I, U>(
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

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> Debug
    for Block<C, F, G, M, N, P>
where
    F: FiniteElement<G, M, N, P>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Block {{ constitutive model: {}, {} elements }}",
            type_name::<C>()
                .rsplit("::")
                .next()
                .unwrap()
                .split("<")
                .next()
                .unwrap(),
            self.elements().len()
        )
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> Elements
    for Block<C, F, G, M, N, P>
where
    F: FiniteElement<G, M, N, P>,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        crate::domain::block::add_node_neighbors(
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
        Self::from((
            constitutive_model,
            PrimitiveConnectivity::from(connectivity),
            coordinates,
        ))
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
        Self::from((
            constitutive_model,
            PrimitiveConnectivity::from(connectivity),
            coordinates,
        ))
    }
}

pub(crate) use crate::domain::block::{finalize_node_neighbors, solver_from_neighbors};
