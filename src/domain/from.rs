use crate::{
    domain::{Blocks, Model, NodalReferenceCoordinates, nodal_coordinates},
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh},
    },
};

/// Builds `Self` from a right-nested list of block connectivities, consuming
/// one connectivity (and its corresponding constitutive model) per level of
/// nesting, from the last connectivity/model out.
///
/// A single block (fem or vem) is the base case, consuming one connectivity
/// directly from its constitutive model `M`. `Blocks<B1, B2>` is the
/// recursive case: it peels the last connectivity off for `B2` (itself a
/// single block) and hands the rest to `B1`, so any right-nested mix of fem
/// and vem blocks, in any number, is buildable this way.
pub(crate) trait FromConnectivities<const D: usize, M>: Sized {
    fn from_connectivities(
        connectivities: Vec<Connectivity>,
        constitutive_models: M,
        coordinates: &NodalReferenceCoordinates<D>,
    ) -> Result<Self, String>;
}

impl<const D: usize, B1, M1, B2, M2> FromConnectivities<D, (M1, M2)> for Blocks<B1, B2>
where
    B1: FromConnectivities<D, M1>,
    B2: FromConnectivities<D, M2>,
{
    fn from_connectivities(
        mut connectivities: Vec<Connectivity>,
        (constitutive_models_1, constitutive_models_2): (M1, M2),
        coordinates: &NodalReferenceCoordinates<D>,
    ) -> Result<Self, String> {
        let last = connectivities
            .pop()
            .ok_or_else(|| "mesh has too few blocks for this model type".to_string())?;
        let block_1 = B1::from_connectivities(connectivities, constitutive_models_1, coordinates)?;
        let block_2 = B2::from_connectivities(vec![last], constitutive_models_2, coordinates)?;
        Ok(Blocks(block_1, block_2))
    }
}

impl<const D: usize, B, M> TryFrom<(Mesh<D>, M)> for Model<B, D>
where
    B: FromConnectivities<D, M>,
{
    type Error = String;
    fn try_from((mesh, constitutive_models): (Mesh<D>, M)) -> Result<Self, Self::Error> {
        let (connectivities, coordinates): (Connectivities, Coordinates<D>) = mesh.into();
        let coordinates = nodal_coordinates(coordinates);
        Ok(Self {
            blocks: B::from_connectivities(
                connectivities.into_members(),
                constitutive_models,
                &coordinates,
            )?,
            coordinates,
        })
    }
}
