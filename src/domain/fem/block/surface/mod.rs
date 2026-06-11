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
    Self: for<'a> From<(C, Connectivity<N>, &'a NodalReferenceCoordinates<3>, Scalar)>,
{
}

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(C, Connectivity<N>, &NodalReferenceCoordinates<3>, Scalar)> for Block<C, F, G, M, N, P>
where
    F: From<(ElementNodalReferenceCoordinates<N>, Scalar)>,
{
    fn from(
        (constitutive_model, connectivity, coordinates, thickness): (
            C,
            Connectivity<N>,
            &NodalReferenceCoordinates<3>,
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
        let connectivity = connectivity.into();
        Self {
            constitutive_model,
            connectivity,
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
