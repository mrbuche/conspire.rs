#[cfg(test)]
mod test;

use crate::fem::{
    ElementModelError, Model, NodalAccelerations, NodalCoordinates,
    mass::InverseMass,
    solid::{NodalForcesSolid, elastic::ElasticElements},
};

impl<B> Model<B, 3>
where
    B: ElasticElements<3>,
{
    pub fn nodal_accelerations(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        external_forces: &NodalForcesSolid<3>,
        masses: &impl InverseMass,
    ) -> Result<NodalAccelerations<3>, ElementModelError> {
        Ok(masses.nodal_accelerations(external_forces, &self.nodal_forces(nodal_coordinates)?))
    }
}
