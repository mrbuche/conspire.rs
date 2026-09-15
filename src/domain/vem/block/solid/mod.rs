pub mod elastic;
pub mod hyperelastic;

use crate::{
    constitutive::solid::Solid,
    domain::solid::SolidElements,
    mechanics::DeformationGradients,
    vem::{
        NodalCoordinates,
        block::{Block, element::solid::SolidVirtualElement},
    },
};

pub type NodalForcesSolid = crate::domain::solid::NodalForcesSolid<3>;
pub type NodalStiffnessesSolid = crate::domain::solid::NodalStiffnessesSolid<3>;
pub type NodalStiffnessesSolidSymmetric = crate::domain::solid::NodalStiffnessesSolidSymmetric<3>;

impl<C, F> SolidElements for Block<C, F>
where
    C: Solid,
    F: SolidVirtualElement,
{
    type DeformationGradients = DeformationGradients;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Vec<Self::DeformationGradients> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.deformation_gradients(Self::element_coordinates(nodal_coordinates, nodes))
            })
            .collect()
    }
}
