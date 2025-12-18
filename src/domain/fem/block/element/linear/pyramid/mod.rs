const G: usize = 5;
const N: usize = 5;

pub type Pyramid = LinearElement<G, N>;

crate::fem::block::element::linear::implement!(Pyramid);

impl Pyramid {
    const fn integration_points() -> [[Scalar; M]; G] {
        [
            [-0.5, 0.0, 1.0 / 6.0],
            [0.5, 0.0, 1.0 / 6.0],
            [0.0, -0.5, 1.0 / 6.0],
            [0.0, 0.5, 1.0 / 6.0],
            [0.0, 0.0, 0.25],
        ]
    }
    const fn integration_weight() -> ScalarList<G> {
        ScalarList::<G>::const_from([5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 16.0 / 27.0])
    }
    const fn reference() -> ElementNodalReferenceCoordinates<N> {
        ElementNodalReferenceCoordinates::<N>::const_from([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    }
    #[cfg(test)]
    const fn shape_functions([xi_1, xi_2, xi_3]: [Scalar; M]) -> [Scalar; N] {
        [
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_3) / 2.0,
        ]
    }
    const fn shape_functions_gradients([xi_1, xi_2, xi_3]: [Scalar; M]) -> [[Scalar; M]; N] {
        [
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
            [0.0, 0.0, 0.5],
        ]
    }
}
