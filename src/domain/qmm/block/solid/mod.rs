pub mod elastic;

use super::{Block, point::solid::SolidElement};
use crate::{
    domain::{NodalCoordinates, NodalVelocities},
    mechanics::{DeformationGradient, DeformationGradientRate},
};

pub use crate::domain::solid::SolidElements;

impl<C> SolidElements for Block<C> {
    type DeformationGradients = DeformationGradient;
    type DeformationGradientRates = DeformationGradientRate;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Vec<Self::DeformationGradients> {
        self.points
            .iter()
            .map(|point| point.deformation_gradients(nodal_coordinates))
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Vec<Self::DeformationGradientRates> {
        self.points
            .iter()
            .map(|point| point.deformation_gradient_rates(nodal_coordinates, nodal_velocities))
            .collect()
    }
}
