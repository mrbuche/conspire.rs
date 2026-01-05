#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        FRAC_SQRT_3_5, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        quadratic::{M, QuadraticElement, QuadraticFiniteElement},
    },
    math::{Scalar, ScalarList},
};

const G: usize = 18;
const N: usize = 15;

pub type Wedge = QuadraticElement<G, N>;

impl FiniteElement<G, M, N> for Wedge {
    fn integration_points() -> ParametricCoordinates<G, M> {
        const ONE_SIXTH: Scalar = 1.0 / 12.0;
        const TWO_THIRDS: Scalar = 2.0 / 3.0;
        [
            [ONE_SIXTH, ONE_SIXTH, FRAC_SQRT_3_5],
            [ONE_SIXTH, TWO_THIRDS, FRAC_SQRT_3_5],
            [TWO_THIRDS, ONE_SIXTH, FRAC_SQRT_3_5],
            [ONE_SIXTH, ONE_SIXTH, -FRAC_SQRT_3_5],
            [ONE_SIXTH, TWO_THIRDS, -FRAC_SQRT_3_5],
            [TWO_THIRDS, ONE_SIXTH, -FRAC_SQRT_3_5],
            [ONE_SIXTH, ONE_SIXTH, 0.0],
            [ONE_SIXTH, TWO_THIRDS, 0.0],
            [TWO_THIRDS, ONE_SIXTH, 0.0],
            [0.5, 0.5, FRAC_SQRT_3_5],
            [0.5, 0.0, FRAC_SQRT_3_5],
            [0.0, 0.5, FRAC_SQRT_3_5],
            [0.5, 0.5, -FRAC_SQRT_3_5],
            [0.5, 0.0, -FRAC_SQRT_3_5],
            [0.0, 0.5, -FRAC_SQRT_3_5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ]
        .into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.0, -1.0],
            [0.5, 0.5, -1.0],
            [0.0, 0.5, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 1.0],
            [0.5, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        const ONE_TWELFTH: Scalar = 1.0 / 12.0;
        const TWO_FIFTEENTHS: Scalar = 2.0 / 15.0;
        const ONE_ONE_HUNDRED_EIGHTH: Scalar = 1.0 / 108.0;
        const TWO_ONE_HUNDRED_THIRTY_FIFTHS: Scalar = 2.0 / 135.0;
        [
            ONE_TWELFTH,
            ONE_TWELFTH,
            ONE_TWELFTH,
            ONE_TWELFTH,
            ONE_TWELFTH,
            ONE_TWELFTH,
            TWO_FIFTEENTHS,
            TWO_FIFTEENTHS,
            TWO_FIFTEENTHS,
            ONE_ONE_HUNDRED_EIGHTH,
            ONE_ONE_HUNDRED_EIGHTH,
            ONE_ONE_HUNDRED_EIGHTH,
            ONE_ONE_HUNDRED_EIGHTH,
            ONE_ONE_HUNDRED_EIGHTH,
            ONE_ONE_HUNDRED_EIGHTH,
            TWO_ONE_HUNDRED_THIRTY_FIFTHS,
            TWO_ONE_HUNDRED_THIRTY_FIFTHS,
            TWO_ONE_HUNDRED_THIRTY_FIFTHS,
        ]
        .into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let xi_0 = 1.0 - xi_1 - xi_2;
        [
            -0.5 * xi_0 * (1.0 - xi_3) * (2.0 * xi_1 + 2.0 * xi_2 + xi_3),
            0.5 * xi_1 * (1.0 - xi_3) * (2.0 * xi_1 - 2.0 - xi_3),
            0.5 * xi_2 * (1.0 - xi_3) * (2.0 * xi_2 - 2.0 - xi_3),
            -0.5 * xi_0 * (1.0 + xi_3) * (2.0 * xi_1 + 2.0 * xi_2 - xi_3),
            0.5 * xi_1 * (1.0 + xi_3) * (2.0 * xi_1 - 2.0 + xi_3),
            0.5 * xi_2 * (1.0 + xi_3) * (2.0 * xi_2 - 2.0 + xi_3),
            2.0 * xi_0 * xi_1 * (1.0 - xi_3),
            2.0 * xi_1 * xi_2 * (1.0 - xi_3),
            2.0 * xi_0 * xi_2 * (1.0 - xi_3),
            xi_0 * (1.0 - xi_3 * xi_3),
            xi_1 * (1.0 - xi_3 * xi_3),
            xi_2 * (1.0 - xi_3 * xi_3),
            2.0 * xi_0 * xi_1 * (1.0 + xi_3),
            2.0 * xi_1 * xi_2 * (1.0 + xi_3),
            2.0 * xi_0 * xi_2 * (1.0 + xi_3),
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let xi_0 = 1.0 - xi_1 - xi_2;
        [
            [
                0.5 * (1.0 - xi_3) * (4.0 * xi_1 + 4.0 * xi_2 + xi_3 - 2.0),
                0.5 * (1.0 - xi_3) * (4.0 * xi_1 + 4.0 * xi_2 + xi_3 - 2.0),
                xi_0 * (xi_1 + xi_2 + xi_3 - 0.5),
            ],
            [
                0.5 * (1.0 - xi_3) * (4.0 * xi_1 - 2.0 - xi_3),
                0.0,
                xi_1 * (xi_3 - xi_1 + 0.5),
            ],
            [
                0.0,
                0.5 * (1.0 - xi_3) * (4.0 * xi_2 - xi_3 - 2.0),
                xi_2 * (xi_3 - xi_2 + 0.5),
            ],
            [
                0.5 * (1.0 + xi_3) * (4.0 * xi_1 + 4.0 * xi_2 - xi_3 - 2.0),
                0.5 * (1.0 + xi_3) * (4.0 * xi_1 + 4.0 * xi_2 - xi_3 - 2.0),
                xi_0 * (-xi_1 - xi_2 + xi_3 + 0.5),
            ],
            [
                0.5 * (1.0 + xi_3) * (4.0 * xi_1 - 2.0 + xi_3),
                0.0,
                xi_1 * (xi_3 + xi_1 - 0.5),
            ],
            [
                0.0,
                0.5 * (1.0 + xi_3) * (4.0 * xi_2 + xi_3 - 2.0),
                xi_2 * (xi_3 + xi_2 - 0.5),
            ],
            [
                2.0 * (1.0 - xi_3) * (1.0 - 2.0 * xi_1 - xi_2),
                -2.0 * xi_1 * (1.0 - xi_3),
                -2.0 * xi_0 * xi_1,
            ],
            [
                2.0 * xi_2 * (1.0 - xi_3),
                2.0 * xi_1 * (1.0 - xi_3),
                -2.0 * xi_1 * xi_2,
            ],
            [
                -2.0 * xi_2 * (1.0 - xi_3),
                2.0 * (1.0 - xi_3) * (xi_0 - xi_2),
                -2.0 * xi_0 * xi_2,
            ],
            [xi_3 * xi_3 - 1.0, xi_3 * xi_3 - 1.0, -2.0 * xi_0 * xi_3],
            [1.0 - xi_3 * xi_3, 0.0, -2.0 * xi_1 * xi_3],
            [0.0, 1.0 - xi_3 * xi_3, -2.0 * xi_2 * xi_3],
            [
                2.0 * (1.0 + xi_3) * (1.0 - 2.0 * xi_1 - xi_2),
                -2.0 * xi_1 * (1.0 + xi_3),
                2.0 * xi_0 * xi_1,
            ],
            [
                2.0 * xi_2 * (1.0 + xi_3),
                2.0 * xi_1 * (1.0 + xi_3),
                2.0 * xi_1 * xi_2,
            ],
            [
                -2.0 * xi_2 * (1.0 + xi_3),
                2.0 * (1.0 + xi_3) * (1.0 - xi_1 - 2.0 * xi_2),
                2.0 * xi_0 * xi_2,
            ],
        ]
        .into()
    }
}

impl QuadraticFiniteElement<G, N> for Wedge {}
