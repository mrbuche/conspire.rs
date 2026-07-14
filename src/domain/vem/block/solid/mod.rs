pub mod elastic;
pub mod hyperelastic;

use crate::{
    mechanics::DeformationGradients,
    vem::{
        NodalCoordinates,
        block::{Block, element::solid::SolidVirtualElement},
    },
};

pub type NodalForcesSolid = crate::fem::solid::NodalForcesSolid<3>;
pub type NodalStiffnessesSolid = crate::fem::solid::NodalStiffnessesSolid<3>;
pub type NodalStiffnessesSolidSymmetric = crate::fem::solid::NodalStiffnessesSolidSymmetric<3>;

pub trait SolidVirtualElements<C, F>
where
    F: SolidVirtualElement,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Vec<DeformationGradients>;
}

impl<C, F> SolidVirtualElements<C, F> for Block<C, F>
where
    F: SolidVirtualElement,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Vec<DeformationGradients> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.deformation_gradients(Self::element_coordinates(nodal_coordinates, nodes))
            })
            .collect()
    }
}
