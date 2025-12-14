pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;
pub mod viscoplastic;

use crate::{
    fem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, GradientVectors,
    },
    math::{Scalars, Tensor},
    mechanics::{DeformationGradientList, DeformationGradientRateList, Forces, Stiffnesses},
};

pub type ElementNodalForcesSolid<const N: usize> = Forces<N>;
pub type ElementNodalStiffnessesSolid<const N: usize> = Stiffnesses<N>;

pub trait SolidFiniteElement<const G: usize, const N: usize> {
    fn deformation_gradients(
        &self,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> DeformationGradientList<G>;
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> DeformationGradientRateList<G>;
    fn gradient_vectors(&self) -> &GradientVectors<G, N>;
    fn integration_weights(&self) -> &Scalars<G>;
}

impl<const G: usize, const N: usize> SolidFiniteElement<G, N> for Element<G, N> {
    fn deformation_gradients(
        &self,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> DeformationGradientList<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_coordinates
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_coordinate, gradient_vector)| {
                        (nodal_coordinate, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        _: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> DeformationGradientRateList<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_velocities
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_velocity, gradient_vector)| {
                        (nodal_velocity, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
    fn gradient_vectors(&self) -> &GradientVectors<G, N> {
        &self.gradient_vectors
    }
    fn integration_weights(&self) -> &Scalars<G> {
        &self.integration_weights
    }
}
