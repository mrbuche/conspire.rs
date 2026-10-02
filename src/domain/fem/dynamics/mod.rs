#[cfg(test)]
mod test;

use crate::fem::{
    ElementModelError, Model, NodalAccelerations, NodalCoordinates,
    block::mass::NodalLumpedMasses,
    mass::LumpedMassElements,
    solid::{NodalForcesSolid, elastic::ElasticElements},
};

impl<B> Model<B, 3>
where
    B: ElasticElements<3> + LumpedMassElements,
{
    pub fn nodal_accelerations(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        external_forces: &NodalForcesSolid<3>,
        nodal_lumped_masses: &NodalLumpedMasses,
    ) -> Result<NodalAccelerations<3>, ElementModelError> {
        Ok(nodal_lumped_masses
            .nodal_accelerations(external_forces, &self.nodal_forces(nodal_coordinates)?))
    }
}
