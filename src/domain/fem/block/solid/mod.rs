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
        NodalCoordinates,
        block::{
            Block,
            element::{FiniteElement, solid::SolidFiniteElement},
        },
        solid::SolidFiniteElements,
    },
    mechanics::DeformationGradientList,
};

pub trait SolidFiniteElementBlock<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    F: SolidFiniteElement<G, M, N, P>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Vec<DeformationGradientList<G>>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    SolidFiniteElementBlock<C, F, G, M, N, P> for Block<C, F, G, M, N, P>
where
    F: SolidFiniteElement<G, M, N, P>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
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

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> SolidFiniteElements
    for Block<C, F, G, M, N, P>
where
    C: Solid,
    F: FiniteElement<G, M, N, P>,
{
}
