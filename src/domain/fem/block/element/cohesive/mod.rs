pub mod elastic;
pub mod linear;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalEitherCoordinates, FiniteElement,
        ShapeFunctionsAtIntegrationPoints, surface::SurfaceFiniteElement,
    },
    math::{ScalarList, Tensor},
    mechanics::CurrentCoordinate,
};
use std::fmt::{self, Debug, Formatter};

pub type Separation = CurrentCoordinate;
pub type Separations<const P: usize> = ElementNodalCoordinates<P>;

const M: usize = 2;

pub struct CohesiveElement<const G: usize, const N: usize, const O: usize> {
    integration_weights: ScalarList<G>,
}

impl<const G: usize, const N: usize, const O: usize> Debug for CohesiveElement<G, N, O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N, O) {
            (3, 6, 1) => "LinearCohesiveWedge",
            (4, 8, 1) => "LinearCohesiveHexahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ G: {G}, N: {N} }}",)
    }
}

impl<const G: usize, const N: usize, const O: usize, const P: usize> SurfaceFiniteElement<G, N, P>
    for CohesiveElement<G, N, O>
where
    Self: FiniteElement<G, M, N, P>,
{
}

pub trait CohesiveFiniteElement<const G: usize, const N: usize, const P: usize>
where
    Self: SurfaceFiniteElement<G, N, P>,
{
    fn nodal_mid_surface<const I: usize>(
        nodal_coordinates: &ElementNodalEitherCoordinates<I, N>,
    ) -> ElementNodalEitherCoordinates<I, P>;
    fn nodal_separations(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<P>;
    fn separations(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<G> {
        Self::shape_functions_at_integration_points()
            .into_iter()
            .map(|shape_functions| {
                Self::nodal_separations(nodal_coordinates)
                    .into_iter()
                    .zip(shape_functions.iter())
                    .map(|(nodal_separation, shape_function)| nodal_separation * shape_function)
                    .sum()
            })
            .collect()
    }
    fn signed_shape_functions() -> ShapeFunctionsAtIntegrationPoints<G, N> {
        Self::shape_functions_at_integration_points()
            .into_iter()
            .map(|shape_functions| {
                shape_functions
                    .iter()
                    .chain(shape_functions.iter())
                    .zip(Self::signs())
                    .map(|(shape_function, sign)| shape_function * sign)
                    .collect()
            })
            .collect()
    }
    fn signs() -> ScalarList<N>;
}
