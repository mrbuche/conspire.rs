use crate::{
    math::{Scalars, Tensor},
    mechanics::{DeformationGradientRates, DeformationGradients, Forces, Stiffnesses},
    vem::block::element::{Element, ElementNodalCoordinates, ElementNodalVelocities},
};

// pub type ElementNodalForcesSolid<const N: usize> = Forces<N>;
// pub type ElementNodalStiffnessesSolid<const N: usize> = Stiffnesses<N>;

pub trait SolidElement {
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
