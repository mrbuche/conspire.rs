use crate::{
    fem::{
        NodalReferenceCoordinates,
        block::{Block, element::ElementNodalReferenceCoordinates},
    },
    math::Scalar,
};

const M: usize = 2;

impl<C, F, const G: usize, const N: usize, const P: usize>
    From<(C, Vec<[usize; N]>, &NodalReferenceCoordinates<3>, Scalar)> for Block<C, F, G, M, N, P>
where
    F: From<(ElementNodalReferenceCoordinates<N>, Scalar)>,
{
    fn from(
        (constitutive_model, connectivity, coordinates, thickness): (
            C,
            Vec<[usize; N]>,
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
