use crate::{
    math::{Scalars, Tensor},
    mechanics::{DeformationGradientRates, DeformationGradients, Forces, Stiffnesses},
    vem::block::element::{Element, ElementNodalCoordinates, ElementNodalVelocities},
};

// pub type ElementNodalForcesSolid<const N: usize> = Forces<N>;
// pub type ElementNodalStiffnessesSolid<const N: usize> = Stiffnesses<N>;

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
    // fn gradient_vectors(&self) -> &GradientVectors<G, N>;
    // fn integration_weights(&self) -> &Scalars<G>;
}

impl SolidVirtualElement for Element {
    fn deformation_gradients(
        &self,
        nodal_coordinates: &ElementNodalCoordinates,
    ) -> DeformationGradients {
        // self.gradient_vectors()
        //     .iter()
        //     .map(|gradient_vectors| {
        //         nodal_coordinates
        //             .iter()
        //             .zip(gradient_vectors.iter())
        //             .map(|(nodal_coordinate, gradient_vector)| {
        //                 (nodal_coordinate, gradient_vector).into()
        //             })
        //             .sum()
        //     })
        //     .collect()
        todo!()
    }
    fn deformation_gradient_rates(
        &self,
        _: &ElementNodalCoordinates,
        nodal_velocities: &ElementNodalVelocities,
    ) -> DeformationGradientRates {
        // self.gradient_vectors()
        //     .iter()
        //     .map(|gradient_vectors| {
        //         nodal_velocities
        //             .iter()
        //             .zip(gradient_vectors.iter())
        //             .map(|(nodal_velocity, gradient_vector)| {
        //                 (nodal_velocity, gradient_vector).into()
        //             })
        //             .sum()
        //     })
        //     .collect()
        todo!()
    }
}
