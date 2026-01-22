#[cfg(test)]
pub mod test;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalEitherCoordinates, ElementNodalReferenceCoordinates,
        FiniteElement, ParametricCoordinate, ParametricCoordinates, ParametricReference,
        ShapeFunctions, ShapeFunctionsGradients,
        cohesive::{CohesiveFiniteElement, M, Separations, linear::LinearCohesiveElement},
        surface::{SurfaceFiniteElement, linear::Triangle as LinearTriangle},
    },
    math::{ScalarList, Tensor},
    mechanics::NormalGradients,
};

// This should share some methods with LinearTriangle<G=3> when get to it.

const G: usize = 3;
const N: usize = 6;
const P: usize = 3;

pub type Wedge = LinearCohesiveElement<G, N>;

impl FiniteElement<G, M, N, P> for Wedge {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [
            [1.0 / 6.0, 1.0 / 6.0],
            [2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0],
        ]
        .into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 6.0; G].into()
    }
    fn scaled_jacobians(nodal_coordinates: &ElementNodalCoordinates<N>) -> ScalarList<P> {
        LinearTriangle::scaled_jacobians(&Self::nodal_mid_surface(nodal_coordinates))
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<P> {
        LinearTriangle::shape_functions(parametric_coordinate)
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, P> {
        LinearTriangle::shape_functions_gradients(parametric_coordinate)
    }
}

impl From<ElementNodalReferenceCoordinates<N>> for Wedge {
    fn from(reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>) -> Self {
        let integration_weights =
            Self::bases(&Self::nodal_mid_surface(&reference_nodal_coordinates))
                .into_iter()
                .zip(Self::parametric_weights())
                .map(|(reference_basis, parametric_weight)| {
                    reference_basis[0].cross(&reference_basis[1]).norm() * parametric_weight
                })
                .collect();
        Self {
            integration_weights,
        }
    }
}

impl CohesiveFiniteElement<G, N, P> for Wedge {
    fn nodal_mid_surface<const I: usize>(
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
    fn nodal_separations(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<P> {
        nodal_coordinates
            .iter()
            .take(P)
            .zip(nodal_coordinates.iter().skip(P))
            .map(|(coordinates_bottom, coordinates_top)| coordinates_top - coordinates_bottom)
            .collect()
    }
    fn normal_gradients_full(
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
    fn signs() -> ScalarList<N> {
        [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0].into()
    }
}
