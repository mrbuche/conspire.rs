#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElementSpecific, ParametricCoordinate,
        ParametricCoordinates, ShapeFunctions, ShapeFunctionsGradients,
        linear::{FRAC_1_SQRT_3, LinearElement, LinearFiniteElement, M},
    },
    math::ScalarList,
};

const G: usize = 8;
const N: usize = 8;

pub type Hexahedron = LinearElement<G, N>;

impl FiniteElementSpecific<G, M, N> for Hexahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        // [
        //     [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
        //     [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3, FRAC_1_SQRT_3],
        //     [-FRAC_1_SQRT_3, FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
        //     [-FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3],
        //     [FRAC_1_SQRT_3, -FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
        //     [FRAC_1_SQRT_3, -FRAC_1_SQRT_3, FRAC_1_SQRT_3],
        //     [FRAC_1_SQRT_3, FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
        //     [FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3],
        // ]
        // .into()
        ParametricCoordinates::<G, M>::const_from([
            [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, -FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, -FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3],
        ])
    }
    fn parametric_reference() -> ElementNodalReferenceCoordinates<N> {
        // [
        //    [-1.0, -1.0, -1.0],
        //     [1.0, -1.0, -1.0],
        //     [1.0, 1.0, -1.0],
        //     [-1.0, 1.0, -1.0],
        //     [-1.0, -1.0, 1.0],
        //     [1.0, -1.0, 1.0],
        //     [1.0, 1.0, 1.0],
        //     [-1.0, 1.0, 1.0],
        // ].into()
        ElementNodalReferenceCoordinates::<N>::const_from([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ])
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0; G].into()
    }
}

impl LinearFiniteElement<G, N> for Hexahedron {
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        // [
        //     (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
        //     (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
        //     (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
        //     (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
        //     (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
        //     (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
        //     (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
        //     (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
        // ].into()
        ShapeFunctions::<N>::const_from([
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
        ])
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        //         [
        //             [
        //                 -(1.0 - xi_2) * (1.0 - xi_3) / 8.0,
        //                 -(1.0 - xi_1) * (1.0 - xi_3) / 8.0,
        //                 -(1.0 - xi_1) * (1.0 - xi_2) / 8.0,
        //             ],
        //             [
        //                 (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
        //                 -(1.0 + xi_1) * (1.0 - xi_3) / 8.0,
        //                 -(1.0 + xi_1) * (1.0 - xi_2) / 8.0,
        //             ],
        //             [
        //                 (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
        //                 (1.0 + xi_1) * (1.0 - xi_3) / 8.0,
        //                 -(1.0 + xi_1) * (1.0 + xi_2) / 8.0,
        //             ],
        //             [
        //                 -(1.0 + xi_2) * (1.0 - xi_3) / 8.0,
        //                 (1.0 - xi_1) * (1.0 - xi_3) / 8.0,
        //                 -(1.0 - xi_1) * (1.0 + xi_2) / 8.0,
        //             ],
        //             [
        //                 -(1.0 - xi_2) * (1.0 + xi_3) / 8.0,
        //                 -(1.0 - xi_1) * (1.0 + xi_3) / 8.0,
        //                 (1.0 - xi_1) * (1.0 - xi_2) / 8.0,
        //             ],
        //             [
        //                 (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
        //                 -(1.0 + xi_1) * (1.0 + xi_3) / 8.0,
        //                 (1.0 + xi_1) * (1.0 - xi_2) / 8.0,
        //             ],
        //             [
        //                 (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
        //                 (1.0 + xi_1) * (1.0 + xi_3) / 8.0,
        //                 (1.0 + xi_1) * (1.0 + xi_2) / 8.0,
        //             ],
        //             [
        //                 -(1.0 + xi_2) * (1.0 + xi_3) / 8.0,
        //                 (1.0 - xi_1) * (1.0 + xi_3) / 8.0,
        //                 (1.0 - xi_1) * (1.0 + xi_2) / 8.0,
        //             ],
        //         ].into()
        ShapeFunctionsGradients::<M, N>::const_from([
            [
                -(1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ],
            [
                -(1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ],
            [
                -(1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ],
            [
                -(1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ],
        ])
    }
}
