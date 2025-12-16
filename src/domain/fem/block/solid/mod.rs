pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use crate::{
    fem::{
        NodalCoordinates,
        block::{Block, element::solid::SolidFiniteElement},
    },
    mechanics::DeformationGradientList,
};

pub use crate::domain::{NodalForcesSolid, NodalStiffnessesSolid};

pub trait SolidFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    F: SolidFiniteElement<G, N>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Vec<DeformationGradientList<G>>;
}

impl<C, F, const G: usize, const N: usize> SolidFiniteElementBlock<C, F, G, N> for Block<C, F, N>
where
    F: SolidFiniteElement<G, N>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Vec<DeformationGradientList<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.deformation_gradients(&Self::element_coordinates(nodal_coordinates, nodes))
            })
            .collect()
    }
}
