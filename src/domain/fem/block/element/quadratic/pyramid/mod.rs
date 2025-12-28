#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        FiniteElement, ParametricCoordinate, ParametricCoordinates, ParametricReference,
        ShapeFunctions, ShapeFunctionsGradients,
        quadratic::{M, QuadraticElement, QuadraticFiniteElement},
    },
    math::{Scalar, ScalarList},
};

const G: usize = 27;
const N: usize = 13;

pub type Pyramid = QuadraticElement<G, N>;

impl FiniteElement<G, M, N> for Pyramid {
    fn integration_points() -> ParametricCoordinates<G, M> {
        integration_points_and_weights().0
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
        integration_points_and_weights().1
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

fn integration_points_and_weights() -> (ParametricCoordinates<G, M>, ScalarList<G>) {
    const X: [f64; 3] = [0.294997790111502, 0.652996233961648, 0.927005975926850];
    const B: [f64; 3] = [0.029950703008581, 0.146246269259866, 0.157136361064887];
    const A_2D: f64 = 0.774596669241483; // sqrt(3/5)
    let u1_2d = [-A_2D, 0.0, A_2D, -A_2D, 0.0, A_2D, -A_2D, 0.0, A_2D];
    let u2_2d = [-A_2D, -A_2D, -A_2D, 0.0, 0.0, 0.0, A_2D, A_2D, A_2D];
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
    let mut points = [[0.0; M]; G];
    let mut weights = [0.0; G];
    let mut i = 0;
    X.into_iter().zip(B).for_each(|(x, b)| {
        u1_2d
            .into_iter()
            .zip(u2_2d)
            .zip(w_2d)
            .for_each(|((u1, u2), w)| {
                points[i][0] = x * u1;
                points[i][1] = x * u2;
                points[i][2] = 1.0 - x;
                weights[i] = w * b;
                i += 1;
            })
    });
    (points.into(), weights.into())
}

impl QuadraticFiniteElement<G, N> for Pyramid {}
