#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalEitherCoordinates, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        linear::{LinearElement, LinearFiniteElement, M},
    },
    math::{ScalarList, Tensor},
};
use std::f64::consts::SQRT_2;

const G: usize = 1;
const N: usize = 4;
const P: usize = N;

const EDGES: usize = 6;

pub type Tetrahedron = LinearElement<G, N>;

impl FiniteElement<G, M, N, P> for Tetrahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [[0.25; M]].into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 6.0; G].into()
    }
    fn scaled_jacobians<const I: usize>(
        nodal_coordinates: ElementNodalEitherCoordinates<I, N>,
    ) -> ScalarList<P> {
        let numerator = ((&nodal_coordinates[1] - &nodal_coordinates[0])
            .cross(&(&nodal_coordinates[2] - &nodal_coordinates[0]))
            * (&nodal_coordinates[3] - &nodal_coordinates[0]))
            * SQRT_2;
        let lengths = lengths(nodal_coordinates);
        [
            numerator / (lengths[0] * lengths[2] * lengths[3]),
            numerator / (lengths[0] * lengths[1] * lengths[4]),
            numerator / (lengths[1] * lengths[2] * lengths[5]),
            numerator / (lengths[3] * lengths[4] * lengths[5]),
        ]
        .into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        [1.0 - xi_1 - xi_2 - xi_3, xi_1, xi_2, xi_3].into()
    }
    fn shape_functions_gradients(
        _parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into()
    }
}

impl LinearFiniteElement<G, N> for Tetrahedron {}

fn edges<const I: usize>(
    nodal_coordinates: ElementNodalEitherCoordinates<I, N>,
) -> ElementNodalEitherCoordinates<I, 6> {
    [
        &nodal_coordinates[1] - &nodal_coordinates[0],
        &nodal_coordinates[2] - &nodal_coordinates[1],
        &nodal_coordinates[0] - &nodal_coordinates[2],
        &nodal_coordinates[3] - &nodal_coordinates[0],
        &nodal_coordinates[3] - &nodal_coordinates[1],
        &nodal_coordinates[3] - &nodal_coordinates[2],
    ]
    .into()
}

fn lengths<const I: usize>(
    nodal_coordinates: ElementNodalEitherCoordinates<I, N>,
) -> ScalarList<EDGES> {
    edges(nodal_coordinates)
        .into_iter()
        .map(|edge| edge.norm())
        .collect()
}
