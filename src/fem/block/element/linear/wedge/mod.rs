const G: usize = 6;
const N: usize = 6;

pub type Wedge = Element<G, N>;

crate::fem::block::element::linear::implement!(Wedge);

use super::SQRT_3_OVER_3;

impl Wedge {
    const fn integration_points() -> [[Scalar; M]; G] {
        [
            [1.0 / 6.0, 1.0 / 6.0, SQRT_3_OVER_3],
            [2.0 / 3.0, 1.0 / 6.0, SQRT_3_OVER_3],
            [1.0 / 6.0, 2.0 / 3.0, -SQRT_3_OVER_3],
            [1.0 / 6.0, 1.0 / 6.0, SQRT_3_OVER_3],
            [2.0 / 3.0, 1.0 / 6.0, SQRT_3_OVER_3],
            [1.0 / 6.0, 2.0 / 3.0, SQRT_3_OVER_3],
        ]
    }
    const fn integration_weight() -> Scalars<G> {
        Scalars::<G>::const_from([1.0 / 6.0; G])
    }
    const fn reference() -> ElementNodalReferenceCoordinates<N> {
        ElementNodalReferenceCoordinates::<N>::const_from([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
    }
    #[cfg(test)]
    const fn shape_functions([xi_1, xi_2, xi_3]: [Scalar; M]) -> [Scalar; N] {
        [
            (1.0 - xi_1 - xi_2) * (1.0 - xi_3) / 2.0,
            xi_1 * (1.0 - xi_3) / 2.0,
            xi_2 * (1.0 - xi_3) / 2.0,
            (1.0 - xi_1 - xi_2) * (1.0 + xi_3) / 2.0,
            xi_1 * (1.0 + xi_3) / 2.0,
            xi_2 * (1.0 + xi_3) / 2.0,
        ]
    }
    const fn shape_functions_gradients([xi_1, xi_2, xi_3]: [Scalar; M]) -> [[Scalar; M]; N] {
        [
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
        ]
    }
}
