#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, GradientVectors, ParametricCoordinate,
        ParametricCoordinates, ShapeFunctions, ShapeFunctionsGradients,
        linear::{FRAC_1_SQRT_3, LinearElement, LinearFiniteElement, M},
    },
    math::ScalarList,
};

const G: usize = 8;
const N: usize = 8;

pub type Hexahedron = LinearElement<G, N>;

// can you impl this more broadly over all linear elements instead? otherwise kind of WET if have to do this way
// and maybe get rid of reset() method, dont think energetic smoothing worth it anymore
impl FiniteElement<G, N> for Hexahedron {
    fn initialize(
        reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>,
    ) -> (GradientVectors<G, N>, ScalarList<G>) {
        let gradient_vectors = Self::shape_functions_gradients_at_integration_points()
            .into_iter()
            .map(|standard_gradient_operator| {
                (&reference_nodal_coordinates * &standard_gradient_operator).inverse_transpose()
                    * standard_gradient_operator
            })
            .collect();
        let integration_weights = Self::shape_functions_gradients_at_integration_points()
            .into_iter()
            .zip(Self::parametric_weights())
            .map(|(standard_gradient_operator, integration_weight)| {
                (&reference_nodal_coordinates * standard_gradient_operator).determinant()
                    * integration_weight
            })
            .collect();
        (gradient_vectors, integration_weights)
    }
    fn reset(&mut self) {
        // let (gradient_vectors, integration_weights) = Self::initialize(Self::reference());
        // self.gradient_vectors = gradient_vectors;
        // self.integration_weights = integration_weights;
        todo!()
    }
}

impl LinearFiniteElement<G, N> for Hexahedron {
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
    fn parametric_weights() -> ScalarList<G> {
        [1.0; G].into()
    }
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
