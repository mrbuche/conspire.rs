pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_viscoplastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use crate::{
    constitutive::solid::Solid,
    fem::{
        NodalCoordinates, NodalVelocities,
        block::{Block, element::solid::SolidFiniteElement},
    },
    mechanics::{DeformationGradientList, DeformationGradientRateList},
};

pub use crate::domain::solid::SolidElements;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> SolidElements
    for Block<C, F, G, M, N, P>
where
    C: Solid,
    F: SolidFiniteElement<G, M, N, P>,
{
    type DeformationGradients = DeformationGradientList<G>;
    type DeformationGradientRates = DeformationGradientRateList<G>;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Vec<Self::DeformationGradients> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.deformation_gradients(&Self::element_coordinates(nodal_coordinates, nodes))
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Vec<Self::DeformationGradientRates> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.deformation_gradient_rates(
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .collect()
    }
}
