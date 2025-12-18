#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, ParametricCoordinate,
        ParametricCoordinates, ShapeFunctions, ShapeFunctionsGradients,
        linear::{LinearElement, LinearFiniteElement, M},
    },
    math::ScalarList,
};

const G: usize = 1;
const N: usize = 4;

pub type Tetrahedron = LinearElement<G, N>;

impl FiniteElement<G, M, N> for Tetrahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        // [
        // [0.25; M],
        // .into()
        ParametricCoordinates::<G, M>::const_from([[0.25; M]])
    }
    fn parametric_reference() -> ElementNodalReferenceCoordinates<N> {
        // [
        // [0.0, 0.0, 0.0],
        // [1.0, 0.0, 0.0],
        // [0.0, 1.0, 0.0],
        // [0.0, 0.0, 1.0],
        // ].into()
        ElementNodalReferenceCoordinates::<N>::const_from([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 6.0; G].into()
    }
}

impl LinearFiniteElement<G, N> for Tetrahedron {
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        // [
        // 1.0 - xi_1 - xi_2 - xi_3, xi_1, xi_2, xi_3
        // ].into()
        ShapeFunctions::<N>::const_from([1.0 - xi_1 - xi_2 - xi_3, xi_1, xi_2, xi_3])
    }
    fn shape_functions_gradients(
        _parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        //         [
        // [-1.0, -1.0, -1.0],
        // [1.0, 0.0, 0.0],
        // [0.0, 1.0, 0.0],
        // [0.0, 0.0, 1.0],
        //         ].into()
        ShapeFunctionsGradients::<M, N>::const_from([
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }
}
