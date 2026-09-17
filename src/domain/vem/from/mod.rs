#[cfg(test)]
mod test;

use crate::{
    domain::{NodalReferenceCoordinates, from::FromConnectivities},
    geometry::mesh::{Connectivity, PolytopalConnectivity},
    vem::block::{Block, element::VirtualElement},
};

impl<C, F> FromConnectivities<3, C> for Block<C, F>
where
    F: VirtualElement,
{
    fn from_connectivities(
        mut connectivities: Vec<Connectivity>,
        constitutive_model: C,
        coordinates: &NodalReferenceCoordinates<3>,
    ) -> Result<Self, String> {
        if connectivities.len() != 1 {
            return Err(format!(
                "mesh has {} blocks, model type expects 1",
                connectivities.len()
            ));
        }
        let connectivity = PolytopalConnectivity::<3>::try_from(connectivities.remove(0))?;
        Ok(Block::from((constitutive_model, connectivity, coordinates)))
    }
}
