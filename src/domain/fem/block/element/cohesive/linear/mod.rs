mod hexahedron;
mod wedge;

pub use hexahedron::Hexahedron;
pub use wedge::Wedge;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalEitherCoordinates, ElementNodalReferenceCoordinates,
        FiniteElement,
        cohesive::{CohesiveElement, Separations},
        surface::SurfaceFiniteElement,
    },
    math::{ScalarList, Tensor},
    mechanics::NormalGradients,
};
use std::iter::repeat_n;

pub type LinearCohesiveElement<const G: usize, const N: usize> = CohesiveElement<G, N, 1>;

pub trait LinearCohesiveFiniteElement<const G: usize, const N: usize, const P: usize>
where
    Self: FiniteElement<G, 2, N, P> + SurfaceFiniteElement<G, N, P>,
{
    fn from_linear(
        reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>,
    ) -> LinearCohesiveElement<G, N> {
        let integration_weights = Self::bases(&Self::nodal_mid_surface_linear(
            &reference_nodal_coordinates,
        ))
        .into_iter()
        .zip(Self::parametric_weights())
        .map(|(reference_basis, parametric_weight)| {
            reference_basis[0].cross(&reference_basis[1]).norm() * parametric_weight
        })
        .collect();
        LinearCohesiveElement {
            integration_weights,
        }
    }
    fn nodal_mid_surface_linear<const I: usize>(
        nodal_coordinates: &ElementNodalEitherCoordinates<I, N>,
    ) -> ElementNodalEitherCoordinates<I, P> {
        nodal_coordinates
            .iter()
            .take(P)
            .zip(nodal_coordinates.iter().skip(P))
            .map(|(coordinates_bottom, coordinates_top)| {
                (coordinates_top + coordinates_bottom) * 0.5
            })
            .collect()
    }
    fn nodal_separations_linear(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<P> {
        nodal_coordinates
            .iter()
            .take(P)
            .zip(nodal_coordinates.iter().skip(P))
            .map(|(coordinates_bottom, coordinates_top)| coordinates_top - coordinates_bottom)
            .collect()
    }
    fn normal_gradients_full_linear(
        nodal_mid_surface: &ElementNodalCoordinates<P>,
    ) -> NormalGradients<N, G> {
        Self::normal_gradients(nodal_mid_surface)
            .into_iter()
            .map(|normal_gradient| {
                normal_gradient
                    .iter()
                    .chain(normal_gradient.iter())
                    .cloned()
                    .collect()
            })
            .collect()
    }
    fn signs_linear() -> ScalarList<N> {
        repeat_n(-1.0, P).chain(repeat_n(1.0, P)).collect()
    }
}
