use super::super::{Block, node::solid::hyperviscoelastic::HyperviscoelasticElement};
use crate::{
    constitutive::{ConstitutiveError, solid::hyperviscoelastic::Hyperviscoelastic},
    domain::{ElementModelError, NodalCoordinates},
    math::Quantity,
    units::Energy,
};

pub use crate::domain::solid::hyperviscoelastic::HyperviscoelasticElements;

impl<C> HyperviscoelasticElements<3> for Block<C>
where
    C: Hyperviscoelastic,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| node.helmholtz_free_energy(&self.constitutive_model, nodal_coordinates))
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
