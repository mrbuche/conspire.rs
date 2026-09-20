pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_plastic;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_viscoplastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use super::Node;
use crate::{
    domain::{NodalCoordinates, NodalVelocities},
    mechanics::{DeformationGradient, DeformationGradientRate},
};

pub use crate::domain::block::element::solid::SolidElement;

impl SolidElement for Node {
    type Coordinates = NodalCoordinates<3>;
    type Velocities = NodalVelocities<3>;
    type DeformationGradients = DeformationGradient;
    type DeformationGradientRates = DeformationGradientRate;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> DeformationGradient {
        self.neighbors()
            .iter()
            .zip(self.gradient_vectors())
            .map(|(&neighbor, gradient_vector)| {
                DeformationGradient::from((&nodal_coordinates[neighbor], gradient_vector))
            })
            .sum()
    }
    fn deformation_gradient_rates(
        &self,
        _nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> DeformationGradientRate {
        self.neighbors()
            .iter()
            .zip(self.gradient_vectors())
            .map(|(&neighbor, gradient_vector)| {
                DeformationGradientRate::from((&nodal_velocities[neighbor], gradient_vector))
            })
            .sum()
    }
}
