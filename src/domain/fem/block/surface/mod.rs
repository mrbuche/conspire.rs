use crate::{
    fem::{
        NodalReferenceCoordinates,
        block::{Block, Connectivity, element::ElementNodalReferenceCoordinates},
    },
    math::Scalar,
};

const M: usize = 2;

pub trait SurfaceFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    Self: From<(C, Connectivity<N>, NodalReferenceCoordinates, Scalar)>,
{
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(C, Connectivity<N>, NodalReferenceCoordinates, Scalar)> for Block<C, F, G, M, N, P>
where
    F: From<(ElementNodalReferenceCoordinates<N>, Scalar)>,
{
    fn from(
        (constitutive_model, connectivity, coordinates, thickness): (
            C,
            Connectivity<N>,
            NodalReferenceCoordinates,
            Scalar,
        ),
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

impl<C, F, const G: usize, const N: usize, const P: usize> SurfaceFiniteElementBlock<C, F, G, N>
    for Block<C, F, G, M, N, P>
where
    F: From<(ElementNodalReferenceCoordinates<N>, Scalar)>,
{
}
