#[cfg(test)]
pub(crate) mod test;

#[cfg(feature = "fem")]
mod density;
#[cfg(feature = "fem")]
pub(crate) mod diffusion;
#[cfg(feature = "fem")]
pub mod electrical;
pub mod element;
#[cfg(feature = "fem")]
pub mod mass;
#[cfg(feature = "fem")]
pub mod multiphysics;
#[cfg(feature = "fem")]
pub mod solid;
#[cfg(feature = "fem")]
pub mod surface;
#[cfg(feature = "fem")]
pub mod thermal;
#[cfg(feature = "fem")]
mod time_scale;
#[cfg(feature = "fem")]
pub mod volume;

#[cfg(feature = "fem")]
use crate::{
    domain::block::{element::Items, trimmed_type_name},
    fem::{
        NodalReferenceCoordinates,
        block::element::{
            ElementNodalReferenceCoordinates, FiniteElement,
            planar::PlanarElementNodalReferenceCoordinates,
        },
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{Quantity, TensorRank1List, TensorRank1Vec},
    units::Volume,
};
#[cfg(feature = "fem")]
use core::fmt::{self, Debug, Formatter};

#[cfg(feature = "fem")]
pub use density::{Densities, DensityField, ElementDensities, NoDensity};
#[cfg(feature = "fem")]
pub use thermal::capacity::{ElementHeatCapacities, HeatCapacities, NoHeatCapacity};

#[cfg(feature = "fem")]
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
}

#[cfg(feature = "fem")]
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

#[cfg(feature = "fem")]
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

#[cfg(feature = "fem")]
impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H> Items
    for Block<C, F, G, M, N, P, R, H>
where
    F: FiniteElement<G, M, N, P>,
{
    fn number_of_items(&self) -> usize {
        self.elements().len()
    }
    fn nodes_of(&self, item: usize) -> &[usize] {
        &self.connectivity().iter().as_slice()[item]
    }
}

#[cfg(feature = "fem")]
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
        }
    }
}

#[cfg(feature = "fem")]
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

#[cfg(feature = "fem")]
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
        }
    }
}

#[cfg(feature = "fem")]
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
