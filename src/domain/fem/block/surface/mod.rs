use crate::{
    fem::{
        NodalReferenceCoordinates,
        block::{Block, Connectivity, element::surface::SurfaceFiniteElementCreation},
    },
    math::Scalar,
};

pub trait SurfaceFiniteElementBlock<C, F, const G: usize, const N: usize> {
    fn new(
        constitutive_model: C,
        connectivity: Connectivity<N>,
        coordinates: NodalReferenceCoordinates,
        thickness: Scalar,
    ) -> Self;
}

impl<C, F, const G: usize, const N: usize> SurfaceFiniteElementBlock<C, F, G, N> for Block<C, F, N>
where
    F: SurfaceFiniteElementCreation<G, N>,
{
    fn new(
        constitutive_model: C,
        connectivity: Connectivity<N>,
        coordinates: NodalReferenceCoordinates,
        thickness: Scalar,
    ) -> Self {
        let elements = connectivity
            .iter()
            .map(|nodes| {
                <F>::from((
                    nodes
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .collect(),
                    thickness,
                ))
            })
            .collect();
        Self {
            constitutive_model,
            connectivity,
            coordinates,
            elements,
        }
    }
}
