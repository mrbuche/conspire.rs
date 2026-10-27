use crate::{
    fem::{
        NodalReferenceCoordinates,
        block::{Block, NoDensity, NoHeatCapacity, element::ElementNodalReferenceCoordinates},
    },
    math::Quantity,
    units::Length,
};

const M: usize = 2;

impl<C, F, const G: usize, const N: usize>
    From<(
        C,
        Vec<[usize; N]>,
        &NodalReferenceCoordinates<3>,
        Quantity<Length>,
    )> for Block<C, F, G, M, N>
where
    F: From<(ElementNodalReferenceCoordinates<N>, Quantity<Length>)>,
{
    fn from(
        (constitutive_model, connectivity, coordinates, thickness): (
            C,
            Vec<[usize; N]>,
            &NodalReferenceCoordinates<3>,
            Quantity<Length>,
        ),
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|nodes| {
                (
                    nodes
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .collect(),
                    thickness,
                )
                    .into()
            })
            .collect();
        let connectivity = connectivity.into();
        Self {
            constitutive_model,
            connectivity,
            elements,
            density: NoDensity,
            heat_capacity: NoHeatCapacity,
        }
    }
}
