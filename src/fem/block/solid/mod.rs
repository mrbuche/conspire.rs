pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use crate::{
    fem::{
        NodalCoordinates,
        block::{
            ElementBlock,
            element::{ElementNodalCoordinates, solid::SolidFiniteElement},
        },
    },
    mechanics::DeformationGradientList,
};

pub trait SolidFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    F: SolidFiniteElement<G, N>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Vec<DeformationGradientList<G>>;
    fn element_nodal_coordinates(
        &self,
        element_connectivity: &[usize; N],
        nodal_coordinates: &NodalCoordinates,
    ) -> ElementNodalCoordinates<N>;
}

impl<C, F, const G: usize, const N: usize> SolidFiniteElementBlock<C, F, G, N>
    for ElementBlock<C, F, N>
where
    F: SolidFiniteElement<G, N>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Vec<DeformationGradientList<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.deformation_gradients(
                    &self.element_nodal_coordinates(element_connectivity, nodal_coordinates),
                )
            })
            .collect()
    }
    fn element_nodal_coordinates(
        &self,
        element_connectivity: &[usize; N],
        nodal_coordinates: &NodalCoordinates,
    ) -> ElementNodalCoordinates<N> {
        element_connectivity
            .iter()
            .map(|&node| nodal_coordinates[node].clone())
            .collect()
    }
}
