pub mod elastic;

use crate::{
    math::Tensor,
    mechanics::{DeformationGradientRates, DeformationGradients, Forces, Stiffnesses},
    vem::block::element::{Element, ElementNodalCoordinates, ElementNodalVelocities},
};

pub type ElementNodalForcesSolid = Forces;
pub type ElementNodalStiffnessesSolid = Stiffnesses;

pub trait SolidVirtualElement {
    fn deformation_gradients<'a>(
        &'a self,
        nodal_coordinates: Vec<Vec<&'a crate::math::TensorRank1<3, 1>>>,
    ) -> DeformationGradients;
}

impl SolidVirtualElement for Element {
    fn deformation_gradients<'a>(
        &'a self,
        nodal_coordinates: Vec<Vec<&'a crate::math::TensorRank1<3, 1>>>,
    ) -> DeformationGradients {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_coordinates
                    .iter()
                    .flatten()
                    .zip(gradient_vectors)
                    .map(|(&nodal_coordinate, gradient_vector)| {
                        (nodal_coordinate, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
}
