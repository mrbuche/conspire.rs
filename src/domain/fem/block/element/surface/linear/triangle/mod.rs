#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, ParametricCoordinate,
        ParametricCoordinates, ShapeFunctions, ShapeFunctionsGradients, surface::SurfaceElement,
    },
    math::ScalarList,
};

const G: usize = 1;
const M: usize = 2;
const N: usize = 3;
const O: usize = 1;

pub type Triangle = SurfaceElement<G, N, O>;

impl FiniteElement<G, M, N> for Triangle {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [[0.25; M]].into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ElementNodalReferenceCoordinates<N> {
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 2.0; G].into()
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
