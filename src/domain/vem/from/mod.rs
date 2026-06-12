#[cfg(test)]
mod test;

use crate::{
    fem::{Model, NodalReferenceCoordinates},
    geometry::mesh::{Connectivities, Mesh, PolytopalConnectivity},
    vem::block::{Block, element::VirtualElement},
};

impl<C, F> TryFrom<(Mesh<3>, C)> for Model<Block<C, F>, 3>
where
    F: VirtualElement,
{
    type Error = String;
    fn try_from((mesh, constitutive_model): (Mesh<3>, C)) -> Result<Self, Self::Error> {
        let (connectivities, coordinates): (Connectivities, NodalReferenceCoordinates<3>) =
            mesh.into();
        let mut connectivities = connectivities.into_members();
        if connectivities.len() != 1 {
            return Err(format!(
                "mesh has {} blocks, model type expects 1",
                connectivities.len()
            ));
        }
        let connectivity = PolytopalConnectivity::<3>::try_from(connectivities.remove(0))?;
        Ok(Model::from((
            Block::from((constitutive_model, connectivity, &coordinates)),
            coordinates,
        )))
    }
}
