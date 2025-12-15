mod hexahedron;
mod pyramid;
mod tetrahedron;

pub use hexahedron::Hexahedron;
pub use pyramid::Pyramid;
pub use tetrahedron::Tetrahedron;

macro_rules! linear_finite_element {
    ($element:ident, $g:expr, $weights:expr) => {
        impl FiniteElement<G, N> for $element {
            fn initialize(
                reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>,
            ) -> (GradientVectors<G, N>, Scalars<G>) {
                let gradient_vectors = Self::standard_gradient_operators()
                    .into_iter()
                    .map(|standard_gradient_operator| {
                        (&reference_nodal_coordinates * &standard_gradient_operator)
                            .inverse_transpose()
                            * standard_gradient_operator
                    })
                    .collect();
                let integration_weights = Self::standard_gradient_operators()
                    .into_iter()
                    .zip(Self::integration_weight())
                    .map(|(standard_gradient_operator, integration_weight)| {
                        (&reference_nodal_coordinates * standard_gradient_operator).determinant()
                            * integration_weight
                    })
                    .collect();
                (gradient_vectors, integration_weights)
            }
            fn reset(&mut self) {
                let (gradient_vectors, integration_weights) = Self::initialize(Self::reference());
                self.gradient_vectors = gradient_vectors;
                self.integration_weights = integration_weights;
            }
        }
        impl $element {
            // const fn integration_point(point: usize) -> [Scalar; M] {
            //     todo!()
            // }
            const fn integration_weight() -> Scalars<G> {
                Scalars::<G>::const_from($weights)
            }
            const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
                StandardGradientOperators::<M, N, P>::const_from([
                    $(
                        Self::shape_functions_gradients(Self::integration_point($g)),
                    )*
                ])
            }
        }
    };
}
pub(crate) use linear_finite_element;
