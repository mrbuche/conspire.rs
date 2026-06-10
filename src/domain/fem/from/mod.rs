#[cfg(test)]
mod test;

use crate::{
    fem::{
        Blocks, ElasticViscoplasticAndElastic, Model, NodalReferenceCoordinates,
        block::{
            Block,
            element::{ElementNodalReferenceCoordinates, FiniteElement},
        },
    },
    geometry::mesh::{Connectivities, Connectivity, Mesh, PrimitiveConnectivity},
};

fn block<C, F, const G: usize, const N: usize, const P: usize>(
    constitutive_model: C,
    connectivity: Connectivity,
    coordinates: &NodalReferenceCoordinates,
) -> Result<Block<C, F, G, 3, N, P>, String>
where
    F: FiniteElement<G, 3, N, P> + From<ElementNodalReferenceCoordinates<N>>,
    PrimitiveConnectivity<3, N>: TryFrom<Connectivity, Error = &'static str>,
{
    Ok(Block::from((
        constitutive_model,
        PrimitiveConnectivity::<3, N>::try_from(connectivity)?
            .into_iter()
            .collect(),
        coordinates,
    )))
}

impl<C, F, const G: usize, const N: usize, const P: usize> TryFrom<(Mesh<3>, C)>
    for Model<Block<C, F, G, 3, N, P>>
where
    F: FiniteElement<G, 3, N, P> + From<ElementNodalReferenceCoordinates<N>>,
    PrimitiveConnectivity<3, N>: TryFrom<Connectivity, Error = &'static str>,
{
    type Error = String;
    fn try_from((mesh, constitutive_model): (Mesh<3>, C)) -> Result<Self, Self::Error> {
        let (connectivities, coordinates): (Connectivities, NodalReferenceCoordinates) =
            mesh.into();
        let mut connectivities = connectivities.into_members();
        if connectivities.len() != 1 {
            return Err(format!(
                "mesh has {} blocks, model type expects 1",
                connectivities.len()
            ));
        }
        Ok(Self {
            blocks: block(constitutive_model, connectivities.remove(0), &coordinates)?,
            coordinates,
        })
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
    for Model<Blocks<Block<C1, F1, G1, 3, N1, P1>, Block<C2, F2, G2, 3, N2, P2>>>
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
        let (connectivities, coordinates): (Connectivities, NodalReferenceCoordinates) =
            mesh.into();
        let mut connectivities = connectivities.into_members().into_iter();
        if connectivities.len() != 2 {
            return Err(format!(
                "mesh has {} blocks, model type expects 2",
                connectivities.len()
            ));
        }
        Ok(Self {
            blocks: Blocks(
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
        let (connectivities, coordinates): (Connectivities, NodalReferenceCoordinates) =
            mesh.into();
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
