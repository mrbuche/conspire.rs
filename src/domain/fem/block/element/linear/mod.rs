mod hexahedron;
mod pyramid;
mod tetrahedron;
mod wedge;

pub use hexahedron::Hexahedron;
pub use pyramid::Pyramid;
pub use tetrahedron::Tetrahedron;
pub use wedge::Wedge;

use crate::{
    fem::block::element::{
        Element, ElementNodalReferenceCoordinates, FiniteElement, ParametricCoordinate,
        ShapeFunctions, ShapeFunctionsAtIntegrationPoints, ShapeFunctionsGradients,
        StandardGradientOperators,
    },
    math::Scalar,
};

const M: usize = 3;

const FRAC_1_SQRT_3: Scalar = 0.577_350_269_189_625_8;

pub type LinearElement<const G: usize, const N: usize> = Element<G, N, 1>;

impl<const G: usize, const N: usize> From<ElementNodalReferenceCoordinates<N>>
    for LinearElement<G, N>
where
    Self: FiniteElement<G, M, N> + LinearFiniteElement<G, N>,
{
    fn from(reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>) -> Self {
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
        Self {
            gradient_vectors,
            integration_weights,
        }
    }
}

pub trait LinearFiniteElement<const G: usize, const N: usize>
where
    Self: FiniteElement<G, M, N>,
{
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N>;
    fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, N> {
        Self::integration_points()
            .into_iter()
            .map(|integration_point| Self::shape_functions(integration_point))
            .collect()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N>;
    fn shape_functions_gradients_at_integration_points() -> StandardGradientOperators<M, N, G> {
        Self::integration_points()
            .into_iter()
            .map(|integration_point| Self::shape_functions_gradients(integration_point))
            .collect()
    }
}
