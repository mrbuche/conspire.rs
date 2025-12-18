#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, ParametricCoordinate,
        ParametricCoordinates, ShapeFunctions, ShapeFunctionsGradients,
        linear::{FRAC_1_SQRT_3, LinearElement, LinearFiniteElement, M},
    },
    math::ScalarList,
};

const G: usize = 6;
const N: usize = 6;

pub type Wedge = LinearElement<G, N>;

impl FiniteElement<G, M, N> for Wedge {
    fn integration_points() -> ParametricCoordinates<G, M> {
        // [
        // [1.0 / 6.0, 1.0 / 6.0, -FRAC_1_SQRT_3],
        // [2.0 / 3.0, 1.0 / 6.0, -FRAC_1_SQRT_3],
        // [1.0 / 6.0, 2.0 / 3.0, -FRAC_1_SQRT_3],
        // [1.0 / 6.0, 1.0 / 6.0, FRAC_1_SQRT_3],
        // [2.0 / 3.0, 1.0 / 6.0, FRAC_1_SQRT_3],
        // [1.0 / 6.0, 2.0 / 3.0, FRAC_1_SQRT_3],
        // .into()
        ParametricCoordinates::<G, M>::const_from([
            [1.0 / 6.0, 1.0 / 6.0, -FRAC_1_SQRT_3],
            [2.0 / 3.0, 1.0 / 6.0, -FRAC_1_SQRT_3],
            [1.0 / 6.0, 2.0 / 3.0, -FRAC_1_SQRT_3],
            [1.0 / 6.0, 1.0 / 6.0, FRAC_1_SQRT_3],
            [2.0 / 3.0, 1.0 / 6.0, FRAC_1_SQRT_3],
            [1.0 / 6.0, 2.0 / 3.0, FRAC_1_SQRT_3],
        ])
    }
    fn parametric_reference() -> ElementNodalReferenceCoordinates<N> {
        // [
        //  [0.0, 0.0, 0.0],
        // [1.0, 0.0, 0.0],
        // [0.0, 1.0, 0.0],
        // [0.0, 0.0, 1.0],
        // [1.0, 0.0, 1.0],
        // [0.0, 1.0, 1.0],
        // ].into()
        ElementNodalReferenceCoordinates::<N>::const_from([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 6.0; G].into()
    }
}

impl LinearFiniteElement<G, N> for Wedge {
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        // [
        // (1.0 - xi_1 - xi_2) * (1.0 - xi_3) / 2.0,
        // xi_1 * (1.0 - xi_3) / 2.0,
        // xi_2 * (1.0 - xi_3) / 2.0,
        // (1.0 - xi_1 - xi_2) * (1.0 + xi_3) / 2.0,
        // xi_1 * (1.0 + xi_3) / 2.0,
        // xi_2 * (1.0 + xi_3) / 2.0,
        // ].into()
        ShapeFunctions::<N>::const_from([
            (1.0 - xi_1 - xi_2) * (1.0 - xi_3) / 2.0,
            xi_1 * (1.0 - xi_3) / 2.0,
            xi_2 * (1.0 - xi_3) / 2.0,
            (1.0 - xi_1 - xi_2) * (1.0 + xi_3) / 2.0,
            xi_1 * (1.0 + xi_3) / 2.0,
            xi_2 * (1.0 + xi_3) / 2.0,
        ])
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        //         [
        // [
        //     -(1.0 - xi_3) / 2.0,
        //     -(1.0 - xi_3) / 2.0,
        //     -(1.0 - xi_1 - xi_2) / 2.0,
        // ],
        // [(1.0 - xi_3) / 2.0, 0.0, -xi_1 / 2.0],
        // [0.0, (1.0 - xi_3) / 2.0, -xi_2 / 2.0],
        // [
        //     -(1.0 + xi_3) / 2.0,
        //     -(1.0 + xi_3) / 2.0,
        //     (1.0 - xi_1 - xi_2) / 2.0,
        // ],
        // [(1.0 + xi_3) / 2.0, 0.0, xi_1 / 2.0],
        // [0.0, (1.0 + xi_3) / 2.0, xi_2 / 2.0],
        //         ].into()
        ShapeFunctionsGradients::<M, N>::const_from([
            [
                -(1.0 - xi_3) / 2.0,
                -(1.0 - xi_3) / 2.0,
                -(1.0 - xi_1 - xi_2) / 2.0,
            ],
            [(1.0 - xi_3) / 2.0, 0.0, -xi_1 / 2.0],
            [0.0, (1.0 - xi_3) / 2.0, -xi_2 / 2.0],
            [
                -(1.0 + xi_3) / 2.0,
                -(1.0 + xi_3) / 2.0,
                (1.0 - xi_1 - xi_2) / 2.0,
            ],
            [(1.0 + xi_3) / 2.0, 0.0, xi_1 / 2.0],
            [0.0, (1.0 + xi_3) / 2.0, xi_2 / 2.0],
        ])
    }
}
