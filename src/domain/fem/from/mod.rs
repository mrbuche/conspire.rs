#[cfg(test)]
mod test;

use crate::{
    domain::from::FromConnectivities,
    fem::{
        ElasticViscoplasticAndElastic, Model, NodalReferenceCoordinates,
        block::{
            Block,
            element::{
                ElementNodalReferenceCoordinates, FiniteElement,
                planar::PlanarElementNodalReferenceCoordinates,
            },
        },
        nodal_coordinates,
    },
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh, PrimitiveConnectivity},
    },
};

fn block<C, F, const D: usize, const G: usize, const N: usize, const P: usize>(
    constitutive_model: C,
    connectivity: Connectivity,
    coordinates: &NodalReferenceCoordinates<D>,
) -> Result<Block<C, F, G, D, N, P>, String>
where
    Block<C, F, G, D, N, P>: for<'a> From<(
        C,
        PrimitiveConnectivity<D, N>,
        &'a NodalReferenceCoordinates<D>,
    )>,
    PrimitiveConnectivity<D, N>: TryFrom<Connectivity, Error = &'static str>,
{
    Ok(Block::from((
        constitutive_model,
        PrimitiveConnectivity::<D, N>::try_from(connectivity)?,
        coordinates,
    )))
}

impl<C, F, const G: usize, const N: usize, const P: usize> FromConnectivities<3, C>
    for Block<C, F, G, 3, N, P>
where
    F: FiniteElement<G, 3, N, P> + From<ElementNodalReferenceCoordinates<N>>,
    PrimitiveConnectivity<3, N>: TryFrom<Connectivity, Error = &'static str>,
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
        block(constitutive_model, connectivities.remove(0), coordinates)
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> FromConnectivities<2, C>
    for Block<C, F, G, 2, N, P>
where
    F: FiniteElement<G, 2, N, P> + From<PlanarElementNodalReferenceCoordinates<N>>,
    PrimitiveConnectivity<2, N>: TryFrom<Connectivity, Error = &'static str>,
{
    fn from_connectivities(
        mut connectivities: Vec<Connectivity>,
        constitutive_model: C,
        coordinates: &NodalReferenceCoordinates<2>,
    ) -> Result<Self, String> {
        if connectivities.len() != 1 {
            return Err(format!(
                "mesh has {} blocks, model type expects 1",
                connectivities.len()
            ));
        }
        block(constitutive_model, connectivities.remove(0), coordinates)
    }
}

impl<
    C1,
    C2,
    F1,
    F2,
    const G1: usize,
    const N1: usize,
    const P1: usize,
    const G2: usize,
    const N2: usize,
    const P2: usize,
> TryFrom<(Mesh<3>, (C1, C2))>
    for Model<
        ElasticViscoplasticAndElastic<Block<C1, F1, G1, 3, N1, P1>, Block<C2, F2, G2, 3, N2, P2>>,
        3,
    >
where
    F1: FiniteElement<G1, 3, N1, P1> + From<ElementNodalReferenceCoordinates<N1>>,
    F2: FiniteElement<G2, 3, N2, P2> + From<ElementNodalReferenceCoordinates<N2>>,
    PrimitiveConnectivity<3, N1>: TryFrom<Connectivity, Error = &'static str>,
    PrimitiveConnectivity<3, N2>: TryFrom<Connectivity, Error = &'static str>,
{
    type Error = String;
    fn try_from(
        (mesh, (constitutive_model_1, constitutive_model_2)): (Mesh<3>, (C1, C2)),
    ) -> Result<Self, Self::Error> {
        let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh.into();
        let coordinates = nodal_coordinates(coordinates);
        let mut connectivities = connectivities.into_members().into_iter();
        if connectivities.len() != 2 {
            return Err(format!(
                "mesh has {} blocks, model type expects 2",
                connectivities.len()
            ));
        }
        Ok(Self {
            blocks: ElasticViscoplasticAndElastic(
                block(
                    constitutive_model_1,
                    connectivities.next().unwrap(),
                    &coordinates,
                )?,
                block(
                    constitutive_model_2,
                    connectivities.next().unwrap(),
                    &coordinates,
                )?,
            ),
            coordinates,
        })
    }
}
