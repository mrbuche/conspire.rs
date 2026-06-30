#[cfg(test)]
mod test;

pub mod cohesive;
pub mod composite;
pub mod linear;
pub mod planar;
pub mod quadratic;
pub mod serendipity;
pub mod solid;
pub mod surface;
pub mod thermal;

use crate::{
    math::{
        Scalar, ScalarList, TensorRank1, TensorRank1List, TensorRank1List2D, TestError,
        defeat_message,
    },
    mechanics::{CoordinateList, CurrentCoordinates, ReferenceCoordinates},
};
use std::fmt::{self, Debug, Display, Formatter};

const A: usize = 9;
const FRAC_1_SQRT_3: Scalar = 0.577_350_269_189_625_8; // nightly feature
const FRAC_SQRT_3_5: Scalar = 0.774_596_669_241_483;

pub type ElementNodalCoordinates<const N: usize> = CurrentCoordinates<N>;
pub type ElementNodalVelocities<const N: usize> = CurrentCoordinates<N>;
pub type ElementNodalEitherCoordinates<const I: usize, const N: usize> = CoordinateList<I, N>;
pub type ElementNodalReferenceCoordinates<const N: usize> = ReferenceCoordinates<N>;
pub type GradientVectors<const D: usize, const G: usize, const N: usize> =
    TensorRank1List2D<D, 0, N, G>;
pub type ParametricCoordinate<const M: usize> = TensorRank1<M, A>;
pub type ParametricCoordinates<const G: usize, const M: usize> = TensorRank1List<M, A, G>;
pub type ParametricReference<const M: usize, const N: usize> = TensorRank1List<M, A, N>;
pub type ShapeFunctions<const N: usize> = TensorRank1<N, A>;
pub type ShapeFunctionsAtIntegrationPoints<const G: usize, const N: usize> =
    TensorRank1List<N, A, G>;
pub type ShapeFunctionsGradients<const M: usize, const N: usize> = TensorRank1List<M, 0, N>;
pub type StandardGradientOperators<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 0, O, P>;
pub type StandardGradientOperatorsTransposed<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 0, P, O>;

pub trait FiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: Clone + Debug,
{
    fn integration_points() -> ParametricCoordinates<G, M>;
    fn integration_weights(&self) -> &ScalarList<G>;
    fn parametric_reference() -> ParametricReference<M, N>;
    fn parametric_weights() -> ScalarList<G>;
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<P>;
    fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, P> {
        Self::integration_points()
            .into_iter()
            .map(|integration_point| Self::shape_functions(integration_point))
            .collect()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, P>;
    fn shape_functions_gradients_at_integration_points() -> StandardGradientOperators<M, P, G> {
        Self::integration_points()
            .into_iter()
            .map(|integration_point| Self::shape_functions_gradients(integration_point))
            .collect()
    }
    fn volume(&self) -> Scalar {
        self.integration_weights().into_iter().sum()
    }
}

#[derive(Clone)]
pub struct Element<const D: usize, const G: usize, const N: usize, const O: usize> {
    gradient_vectors: GradientVectors<D, G, N>,
    integration_weights: ScalarList<G>,
}

impl<const D: usize, const G: usize, const N: usize, const O: usize> Element<D, G, N, O> {
    fn gradient_vectors(&self) -> &GradientVectors<D, G, N> {
        &self.gradient_vectors
    }
}

impl<const D: usize, const G: usize, const N: usize, const O: usize> Debug for Element<D, G, N, O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (D, G, N, O) {
            (2, 1, 3, 1) => "LinearTriangle",
            (2, 4, 4, 1) => "LinearQuadrilateral",
            (3, 8, 8, 1) => "LinearHexahedron",
            (3, 8, 5, 1) => "LinearPyramid",
            (3, 1, 4, 1) => "LinearTetrahedron",
            (3, 6, 6, 1) => "LinearWedge",
            (3, 27, 27, 2) => "QuadraticHexahedron",
            (3, 4, 10, 2) => "QuadraticTetrahedron",
            (3, 27, 13, 2) => "QuadraticPyramid",
            (3, 18, 15, 2) => "QuadraticWedge",
            (3, 27, 20, 2) => "SerendipityHexahedron",
            (3, 4, 10, 0) => "CompositeTetrahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ integration points: {G}, nodes: {N} }}",)
    }
}

fn basic_from<const D: usize, const G: usize, const N: usize, const O: usize>(
    reference_nodal_coordinates: TensorRank1List<D, 0, N>,
) -> Element<D, G, N, O>
where
    Element<D, G, N, O>: FiniteElement<G, D, N, N>,
{
    let gradient_vectors = Element::shape_functions_gradients_at_integration_points()
        .into_iter()
        .map(|standard_gradient_operator| {
            (&reference_nodal_coordinates * &standard_gradient_operator).inverse_transpose()
                * standard_gradient_operator
        })
        .collect();
    let integration_weights = Element::shape_functions_gradients_at_integration_points()
        .into_iter()
        .zip(Element::parametric_weights())
        .map(|(standard_gradient_operator, integration_weight)| {
            (&reference_nodal_coordinates * standard_gradient_operator).determinant()
                * integration_weight
        })
        .collect();
    Element {
        gradient_vectors,
        integration_weights,
    }
}

pub enum FiniteElementError {
    Upstream(String, String),
}

impl From<FiniteElementError> for TestError {
    fn from(error: FiniteElementError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl Debug for FiniteElementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, element) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In finite element: {element}."
                )
            }
        };
        write!(f, "\n{error}\n\x1b[0;2;31m{}\x1b[0m\n", defeat_message())
    }
}

impl Display for FiniteElementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, element) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In finite element: {element}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}
