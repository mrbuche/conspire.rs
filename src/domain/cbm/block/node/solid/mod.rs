pub mod elastic;

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
        self.gradient_vectors()
            .iter()
            .map(|(neighbor, gradient_vector)| {
                DeformationGradient::from((&nodal_coordinates[*neighbor], gradient_vector))
            })
            .sum()
    }
    fn deformation_gradient_rates(
        &self,
        _nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> DeformationGradientRate {
        self.gradient_vectors()
            .iter()
            .map(|(neighbor, gradient_vector)| {
                DeformationGradientRate::from((&nodal_velocities[*neighbor], gradient_vector))
            })
            .sum()
    }
}
