#[cfg(test)]
pub mod test;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        surface::{M, linear::LinearSurfaceElement},
    },
    math::{ScalarList, Tensor},
};
use std::f64::consts::FRAC_PI_3;

// When implement G=3, share the methods with cohesive linear wedge.

const G: usize = 1;
const N: usize = 3;
const P: usize = N;

pub type Triangle = LinearSurfaceElement<G, N>;

impl FiniteElement<G, M, N, P> for Triangle {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [[1.0 / 3.0; M]].into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]].into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 2.0; G].into()
    }
    fn scaled_jacobians(nodal_coordinates: &ElementNodalCoordinates<N>) -> ScalarList<P> {
        let sin_60 = FRAC_PI_3.sin();
        angles(nodal_coordinates)
            .into_iter()
            .map(|angle| angle.sin() / sin_60)
            .collect()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2] = parametric_coordinate.into();
        [1.0 - xi_1 - xi_2, xi_1, xi_2].into()
    }
    fn shape_functions_gradients(
        _parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]].into()
    }
}

fn angles(nodal_coordinates: &ElementNodalCoordinates<P>) -> ScalarList<P> {
    let l0 = (&nodal_coordinates[2] - &nodal_coordinates[1]).normalized();
    let l1 = (&nodal_coordinates[0] - &nodal_coordinates[2]).normalized();
    let l2 = (&nodal_coordinates[1] - &nodal_coordinates[0]).normalized();
    [(-&l0 * &l1).acos(), (-l1 * &l2).acos(), (-l2 * l0).acos()].into()
}
