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
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    mechanics::DeformationGradientList,
};

pub trait SolidFiniteElementBlock<C, F, const G: usize, const M: usize, const N: usize>
where
    F: SolidFiniteElement<G, M, N>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Vec<DeformationGradientList<G>>;
}

impl<C, F, const G: usize, const M: usize, const N: usize> SolidFiniteElementBlock<C, F, G, M, N>
    for Block<C, F, G, M, N>
where
    F: SolidFiniteElement<G, M, N>,
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
