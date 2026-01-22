#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, FRAC_1_SQRT_3, FiniteElement, ParametricCoordinate,
        ParametricCoordinates, ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        surface::{M, linear::LinearSurfaceElement},
    },
    math::ScalarList,
};

const G: usize = 4;
const N: usize = 4;
const P: usize = N;

pub type Quadrilateral = LinearSurfaceElement<G, N>;

impl FiniteElement<G, M, N, P> for Quadrilateral {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [
            [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, FRAC_1_SQRT_3],
        ]
        .into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]].into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0; G].into()
    }
    fn scaled_jacobians(_nodal_coordinates: &ElementNodalCoordinates<N>) -> ScalarList<P> {
        todo!()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2] = parametric_coordinate.into();
        [
            (1.0 - xi_1) * (1.0 - xi_2) / 4.0,
            (1.0 + xi_1) * (1.0 - xi_2) / 4.0,
            (1.0 + xi_1) * (1.0 + xi_2) / 4.0,
            (1.0 - xi_1) * (1.0 + xi_2) / 4.0,
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2] = parametric_coordinate.into();
        [
            [-(1.0 - xi_2) / 4.0, -(1.0 - xi_1) / 4.0],
            [(1.0 - xi_2) / 4.0, -(1.0 + xi_1) / 4.0],
            [(1.0 + xi_2) / 4.0, (1.0 + xi_1) / 4.0],
            [-(1.0 + xi_2) / 4.0, (1.0 - xi_1) / 4.0],
        ]
        .into()
    }
}
