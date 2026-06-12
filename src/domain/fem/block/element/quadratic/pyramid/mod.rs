#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        FRAC_SQRT_3_5, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        quadratic::{M, QuadraticElement, QuadraticFiniteElement},
    },
    math::{Scalar, ScalarList, TensorRank1},
};

const G: usize = 27;
const N: usize = 13;
const P: usize = N;

pub type Pyramid = QuadraticElement<G, N>;

impl FiniteElement<G, M, N, P> for Pyramid {
    fn integration_points() -> ParametricCoordinates<G, M> {
        const X: [f64; 3] = [0.294997790111502, 0.652996233961648, 0.927005975926850];
        let u1_2d = [
            -FRAC_SQRT_3_5,
            0.0,
            FRAC_SQRT_3_5,
            -FRAC_SQRT_3_5,
            0.0,
            FRAC_SQRT_3_5,
            -FRAC_SQRT_3_5,
            0.0,
            FRAC_SQRT_3_5,
        ];
        let u2_2d = [
            -FRAC_SQRT_3_5,
            -FRAC_SQRT_3_5,
            -FRAC_SQRT_3_5,
            0.0,
            0.0,
            0.0,
            FRAC_SQRT_3_5,
            FRAC_SQRT_3_5,
            FRAC_SQRT_3_5,
        ];
        X.into_iter()
            .flat_map(|x| {
                u1_2d
                    .into_iter()
                    .zip(u2_2d)
                    .map(move |(u1, u2)| TensorRank1::from([x * u1, x * u2, 1.0 - x]))
            })
            .collect()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        const B: [f64; 3] = [0.029950703008581, 0.146246269259866, 0.157136361064887];
        const W1: f64 = 5.0 / 9.0;
        const W2: f64 = 8.0 / 9.0;
        let w_2d = [
            W1 * W1,
            W2 * W1,
            W1 * W1,
            W1 * W2,
            W2 * W2,
            W1 * W2,
            W1 * W1,
            W2 * W1,
            W1 * W1,
        ];
        B.into_iter()
            .flat_map(|b| w_2d.into_iter().map(move |w| w * b))
            .collect()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let bottom = bottom(xi_3);
        [
            0.25 * (-xi_1 - xi_2 - 1.0)
                * ((1.0 - xi_1) * (1.0 - xi_2) - xi_3 + xi_1 * xi_2 * xi_3 / bottom),
            0.25 * (xi_1 - xi_2 - 1.0)
                * ((1.0 + xi_1) * (1.0 - xi_2) - xi_3 - xi_1 * xi_2 * xi_3 / bottom),
            0.25 * (xi_1 + xi_2 - 1.0)
                * ((1.0 + xi_1) * (1.0 + xi_2) - xi_3 + xi_1 * xi_2 * xi_3 / bottom),
            0.25 * (-xi_1 + xi_2 - 1.0)
                * ((1.0 - xi_1) * (1.0 + xi_2) - xi_3 - xi_1 * xi_2 * xi_3 / bottom),
            xi_3 * (2.0 * xi_3 - 1.0),
            0.5 * (1.0 + xi_1 - xi_3) * (1.0 - xi_1 - xi_3) * (1.0 - xi_2 - xi_3) / bottom,
            0.5 * (1.0 + xi_2 - xi_3) * (1.0 - xi_2 - xi_3) * (1.0 + xi_1 - xi_3) / bottom,
            0.5 * (1.0 + xi_1 - xi_3) * (1.0 - xi_1 - xi_3) * (1.0 + xi_2 - xi_3) / bottom,
            0.5 * (1.0 + xi_2 - xi_3) * (1.0 - xi_2 - xi_3) * (1.0 - xi_1 - xi_3) / bottom,
            xi_3 * (1.0 - xi_1 - xi_3) * (1.0 - xi_2 - xi_3) / bottom,
            xi_3 * (1.0 + xi_1 - xi_3) * (1.0 - xi_2 - xi_3) / bottom,
            xi_3 * (1.0 + xi_1 - xi_3) * (1.0 + xi_2 - xi_3) / bottom,
            xi_3 * (1.0 - xi_1 - xi_3) * (1.0 + xi_2 - xi_3) / bottom,
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        let bottom = bottom(xi_3);
        let bottom_squared = bottom * bottom;
        [
            [
                0.25 * ((-xi_1 - xi_2 - 1.0) * (-1.0 + xi_2 + xi_2 * xi_3 / bottom)
                    - ((1.0 - xi_1) * (1.0 - xi_2) - xi_3 + xi_1 * xi_2 * xi_3 / bottom)),
                0.25 * ((-xi_1 - xi_2 - 1.0) * (-1.0 + xi_1 + xi_1 * xi_3 / bottom)
                    - ((1.0 - xi_1) * (1.0 - xi_2) - xi_3 + xi_1 * xi_2 * xi_3 / bottom)),
                0.25 * (-xi_1 - xi_2 - 1.0)
                    * (-1.0 + xi_1 * xi_2 / bottom + xi_1 * xi_2 * xi_3 / bottom_squared),
            ],
            [
                0.25 * ((xi_1 - xi_2 - 1.0) * (1.0 - xi_2 - xi_2 * xi_3 / bottom)
                    + ((1.0 + xi_1) * (1.0 - xi_2) - xi_3 - xi_1 * xi_2 * xi_3 / bottom)),
                0.25 * ((xi_1 - xi_2 - 1.0) * (-1.0 - xi_1 - xi_1 * xi_3 / bottom)
                    - ((1.0 + xi_1) * (1.0 - xi_2) - xi_3 - xi_1 * xi_2 * xi_3 / bottom)),
                0.25 * (xi_1 - xi_2 - 1.0)
                    * (-1.0 - xi_1 * xi_2 / bottom - xi_1 * xi_2 * xi_3 / bottom_squared),
            ],
            [
                0.25 * ((xi_1 + xi_2 - 1.0) * (1.0 + xi_2 + xi_2 * xi_3 / bottom)
                    + ((1.0 + xi_1) * (1.0 + xi_2) - xi_3 + xi_1 * xi_2 * xi_3 / bottom)),
                0.25 * ((xi_1 + xi_2 - 1.0) * (1.0 + xi_1 + xi_1 * xi_3 / bottom)
                    + ((1.0 + xi_1) * (1.0 + xi_2) - xi_3 + xi_1 * xi_2 * xi_3 / bottom)),
                0.25 * (xi_1 + xi_2 - 1.0)
                    * (-1.0 + xi_1 * xi_2 / bottom + xi_1 * xi_2 * xi_3 / bottom_squared),
            ],
            [
                0.25 * ((-xi_1 + xi_2 - 1.0) * (-1.0 - xi_2 - xi_2 * xi_3 / bottom)
                    - ((1.0 - xi_1) * (1.0 + xi_2) - xi_3 - xi_1 * xi_2 * xi_3 / bottom)),
                0.25 * ((-xi_1 + xi_2 - 1.0) * (1.0 - xi_1 - xi_1 * xi_3 / bottom)
                    + ((1.0 - xi_1) * (1.0 + xi_2) - xi_3 - xi_1 * xi_2 * xi_3 / bottom)),
                0.25 * (-xi_1 + xi_2 - 1.0)
                    * (-1.0 - xi_1 * xi_2 / bottom - xi_1 * xi_2 * xi_3 / bottom_squared),
            ],
            [0.0, 0.0, 4.0 * xi_3 - 1.0],
            [
                -xi_1 * (1.0 - xi_2 - xi_3) / bottom,
                -0.5 * (1.0 - xi_1 - xi_3) * (1.0 + xi_1 - xi_3) / bottom,
                0.5 * (xi_1 * xi_1 * xi_2 / bottom_squared + xi_2) - 1.0 + xi_3,
            ],
            [
                0.5 * (1.0 - xi_2 - xi_3) * (1.0 + xi_2 - xi_3) / bottom,
                -xi_2 * (1.0 + xi_1 - xi_3) / bottom,
                -0.5 * (xi_1 * xi_2 * xi_2 / bottom_squared + xi_1) - 1.0 + xi_3,
            ],
            [
                -xi_1 * (1.0 + xi_2 - xi_3) / bottom,
                0.5 * (1.0 - xi_1 - xi_3) * (1.0 + xi_1 - xi_3) / bottom,
                -0.5 * (xi_1 * xi_1 * xi_2 / bottom_squared + xi_2) - 1.0 + xi_3,
            ],
            [
                -0.5 * (1.0 - xi_2 - xi_3) * (1.0 + xi_2 - xi_3) / bottom,
                -xi_2 * (1.0 - xi_1 - xi_3) / bottom,
                0.5 * (xi_1 * xi_2 * xi_2 / bottom_squared + xi_1) - 1.0 + xi_3,
            ],
            [
                -(1.0 - xi_2 - xi_3) * xi_3 / bottom,
                -(1.0 - xi_1 - xi_3) * xi_3 / bottom,
                xi_1 * xi_2 / bottom_squared + 1.0 - xi_1 - xi_2 - 2.0 * xi_3,
            ],
            [
                (1.0 - xi_2 - xi_3) * xi_3 / bottom,
                -(1.0 + xi_1 - xi_3) * xi_3 / bottom,
                -xi_1 * xi_2 / bottom_squared + 1.0 + xi_1 - xi_2 - 2.0 * xi_3,
            ],
            [
                (1.0 + xi_2 - xi_3) * xi_3 / bottom,
                (1.0 + xi_1 - xi_3) * xi_3 / bottom,
                xi_1 * xi_2 / bottom_squared + 1.0 + xi_1 + xi_2 - 2.0 * xi_3,
            ],
            [
                -(1.0 + xi_2 - xi_3) * xi_3 / bottom,
                (1.0 - xi_1 - xi_3) * xi_3 / bottom,
                -xi_1 * xi_2 / bottom_squared + 1.0 - xi_1 + xi_2 - 2.0 * xi_3,
            ],
        ]
        .into()
    }
}

fn bottom(xi_3: Scalar) -> Scalar {
    const SMALL: Scalar = 4e1 * f64::EPSILON;
    if (1.0 - xi_3).abs() > SMALL {
        1.0 - xi_3
    } else {
        SMALL
    }
}

impl QuadraticFiniteElement<G, N> for Pyramid {}
