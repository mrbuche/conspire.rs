mod hexahedron;
mod tetrahedron;

pub use hexahedron::Hexahedron;
pub use tetrahedron::Tetrahedron;

macro_rules! linear_finite_element {
    ($element:ident) => {
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
                    .map(|standard_gradient_operator| {
                        (&reference_nodal_coordinates * standard_gradient_operator).determinant()
                            * Self::integration_weight()
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
    };
}
pub(crate) use linear_finite_element;
