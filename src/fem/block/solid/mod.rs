pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use crate::{
    fem::{
        NodalCoordinatesBlock,
        block::{ElementBlock, element::SolidFiniteElement},
    },
    mechanics::DeformationGradientList,
};

pub trait SolidFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    F: SolidFiniteElement<G, N>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Vec<DeformationGradientList<G>>;
}

impl<C, F, const G: usize, const N: usize> SolidFiniteElementBlock<C, F, G, N>
    for ElementBlock<C, F, N>
where
    F: SolidFiniteElement<G, N>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinatesBlock,
    ) -> Vec<DeformationGradientList<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.deformation_gradients(
                    &self.nodal_coordinates_element(element_connectivity, nodal_coordinates),
                )
            })
            .collect()
    }
}
