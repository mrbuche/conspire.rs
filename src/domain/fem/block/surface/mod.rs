use crate::{
    fem::{
        NodalReferenceCoordinates,
        block::{Block, Connectivity, element::surface::SurfaceFiniteElement},
    },
    math::Scalar,
};

pub trait SurfaceFiniteElementBlock<C, F, const G: usize, const N: usize, const P: usize>
where
    F: SurfaceFiniteElement<G, N, P>,
{
    fn new(
        constitutive_model: C,
        connectivity: Connectivity<N>,
        coordinates: NodalReferenceCoordinates,
        thickness: Scalar,
    ) -> Self;
}

impl<C, F, const G: usize, const N: usize, const P: usize> SurfaceFiniteElementBlock<C, F, G, N, P>
    for Block<C, F, N>
where
    F: SurfaceFiniteElement<G, N, P>,
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
                <F>::new(
                    nodes
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .collect(),
                    thickness,
                )
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
