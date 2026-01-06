pub mod elastic;
pub mod hyperelastic;

use crate::{
    math::Tensor,
    mechanics::{DeformationGradients, Forces, Stiffnesses},
    vem::block::element::{Element, ElementNodalCoordinates, VirtualElement},
};

pub type ElementNodalForcesSolid = Forces;
pub type ElementNodalStiffnessesSolid = Stiffnesses;

pub trait SolidVirtualElement {
    fn deformation_gradients<'a>(
        &'a self,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> DeformationGradients;
}

impl SolidVirtualElement for Element {
    fn deformation_gradients<'a>(
        &'a self,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> DeformationGradients {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_coordinates
                    .iter()
                    .zip(gradient_vectors)
                    .map(|(&nodal_coordinate, gradient_vector)| {
                        (nodal_coordinate, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
}
