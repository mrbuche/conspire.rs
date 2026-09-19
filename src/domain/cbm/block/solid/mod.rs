pub mod elastic;
pub mod elastic_hyperviscous;
pub mod hyperelastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;

use super::{Block, node::solid::SolidElement};
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
        self.nodes
            .iter()
            .map(|node| node.deformation_gradients(nodal_coordinates))
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Vec<Self::DeformationGradientRates> {
        self.nodes
            .iter()
            .map(|node| node.deformation_gradient_rates(nodal_coordinates, nodal_velocities))
            .collect()
    }
}
