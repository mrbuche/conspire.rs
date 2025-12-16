use crate::{
    mechanics::DeformationGradients,
    vem::{
        NodalCoordinates,
        block::{Block, element::solid::SolidVirtualElement},
    },
};

pub use crate::domain::{NodalForcesSolid, NodalStiffnessesSolid};

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
            .zip(self.element_faces())
            .map(|(element, faces)| {
                element.deformation_gradients(&Self::element_coordinates(
                    nodal_coordinates,
                    faces,
                    self.face_nodes(),
                ))
            })
            .collect()
    }
}
