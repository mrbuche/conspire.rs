pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;
pub mod viscoplastic;

use crate::{
    fem::{GradientVectors, NodalCoordinates, NodalVelocities},
    math::{Scalars, Tensor},
    mechanics::{DeformationGradientList, DeformationGradientRateList},
};
use std::fmt::{self, Debug, Formatter};

// pub type SolidElement<const G: usize, const N: usize> = Element<G, GradientVectors<G, N>>;

pub struct SolidElement<const G: usize, const N: usize> {
    pub gradient_vectors: GradientVectors<G, N>,
    pub integration_weights: Scalars<G>,
    // gradient_vectors: GradientVectors<G, N>,
    // integration_weights: Scalars<G>,
}

impl<const G: usize, const N: usize> Debug for SolidElement<G, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N) {
            (1, 4) => "LinearTetrahedron",
            (8, 8) => "LinearHexahedron",
            (4, 10) => "CompositeTetrahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ Solid, G: {G}, N: {N} }}",)
    }
}

pub trait SolidFiniteElement<const G: usize, const N: usize> {
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradientList<G>;
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRateList<G>;
    fn gradient_vectors(&self) -> &GradientVectors<G, N>;
    fn integration_weights(&self) -> &Scalars<G>;
}

impl<const G: usize, const N: usize> SolidFiniteElement<G, N> for SolidElement<G, N> {
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
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
        _: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
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
