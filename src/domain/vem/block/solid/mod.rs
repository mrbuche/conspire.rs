pub mod elastic;
pub mod hyperelastic;

use crate::{
    mechanics::DeformationGradients,
    vem::{
        NodalCoordinates,
        block::{Block, element::solid::SolidVirtualElement},
    },
};

pub use crate::fem::solid::{NodalForcesSolid, NodalStiffnessesSolid};

pub trait SolidVirtualElementBlock<C, F>
where
    F: SolidVirtualElement,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Vec<DeformationGradients>;
}

impl<C, F> SolidVirtualElementBlock<C, F> for Block<C, F>
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
