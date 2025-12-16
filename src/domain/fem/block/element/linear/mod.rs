mod hexahedron;
mod pyramid;
mod tetrahedron;
mod wedge;

pub use hexahedron::Hexahedron;
pub use pyramid::Pyramid;
pub use tetrahedron::Tetrahedron;
pub use wedge::Wedge;

use crate::math::Scalar;

const FRAC_1_SQRT_3: Scalar = 0.577_350_269_189_625_8;

macro_rules! implement {
    ($element:ident) => {
        #[cfg(test)]
        mod test;
        #[cfg(test)]
        use crate::fem::block::element::ShapeFunctionsAtIntegrationPoints;
        use crate::{
            fem::block::element::{
                Element, ElementNodalReferenceCoordinates, FiniteElement, GradientVectors,
                StandardGradientOperators,
            },
            math::{Scalar, Scalars},
        };
        const M: usize = 3;
        const P: usize = G;
        #[cfg(test)]
        const Q: usize = N;
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
            #[cfg(test)]
            const fn shape_functions_at_integration_points()
            -> ShapeFunctionsAtIntegrationPoints<G, Q> {
                let mut g = 0;
                let mut shape_functions = [[0.0; Q]; G];
                let integration_points = Self::integration_points();
                while g < G {
                    shape_functions[g] = Self::shape_functions(integration_points[g]);
                    g += 1;
                }
                ShapeFunctionsAtIntegrationPoints::<G, Q>::const_from(shape_functions)
            }
            const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
                let mut g = 0;
                let mut shape_functions_gradients = [[[0.0; M]; N]; G];
                let integration_points = Self::integration_points();
                while g < G {
                    shape_functions_gradients[g] =
                        Self::shape_functions_gradients(integration_points[g]);
                    g += 1;
                }
                StandardGradientOperators::<M, N, P>::const_from(shape_functions_gradients)
            }
        }
    };
}
pub(crate) use implement;
