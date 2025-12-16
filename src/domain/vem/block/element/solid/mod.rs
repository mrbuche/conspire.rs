pub mod elastic;

use crate::{
    math::Tensor,
    mechanics::{DeformationGradientRates, DeformationGradients, Forces, Stiffnesses},
    vem::block::element::{Element, ElementNodalCoordinates, ElementNodalVelocities},
};

pub type ElementNodalForcesSolid = Forces;
pub type ElementNodalStiffnessesSolid = Stiffnesses;

pub trait SolidVirtualElement {
    fn deformation_gradients(
        &self,
        nodal_coordinates: &ElementNodalCoordinates,
    ) -> DeformationGradients;
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &ElementNodalCoordinates,
        nodal_velocities: &ElementNodalVelocities,
    ) -> DeformationGradientRates;
}

impl SolidVirtualElement for Element {
    fn deformation_gradients(
        &self,
        nodal_coordinates: &ElementNodalCoordinates,
    ) -> DeformationGradients {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_coordinates
                    .iter()
                    .flatten()
                    .zip(gradient_vectors)
                    .map(|(nodal_coordinate, gradient_vector)| {
                        (nodal_coordinate, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        _: &ElementNodalCoordinates,
        nodal_velocities: &ElementNodalVelocities,
    ) -> DeformationGradientRates {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_velocities
                    .iter()
                    .flatten()
                    .zip(gradient_vectors)
                    .map(|(nodal_velocity, gradient_vector)| {
                        (nodal_velocity, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
}
