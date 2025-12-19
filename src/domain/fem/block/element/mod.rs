#[cfg(test)]
mod test;

pub mod composite;
pub mod linear;
pub mod solid;
pub mod surface;
pub mod thermal;

use crate::{
    defeat_message,
    math::{ScalarList, TensorRank1, TensorRank1List, TensorRank1List2D, TestError},
    mechanics::{CurrentCoordinates, ReferenceCoordinates, VectorList2D},
};
use std::fmt::{self, Debug, Display, Formatter};

const A: usize = 9;

pub type ElementNodalCoordinates<const N: usize> = CurrentCoordinates<N>;
pub type ElementNodalVelocities<const N: usize> = CurrentCoordinates<N>;
pub type ElementNodalReferenceCoordinates<const N: usize> = ReferenceCoordinates<N>;
pub type GradientVectors<const G: usize, const N: usize> = VectorList2D<0, N, G>;
pub type ParametricCoordinate<const M: usize> = TensorRank1<M, A>;
pub type ParametricCoordinates<const G: usize, const M: usize> = TensorRank1List<M, A, G>;
pub type ShapeFunctions<const N: usize> = TensorRank1<N, A>;
pub type ShapeFunctionsAtIntegrationPoints<const G: usize, const N: usize> =
    TensorRank1List<N, A, G>;
pub type ShapeFunctionsGradients<const M: usize, const N: usize> = TensorRank1List<M, 0, N>;
pub type StandardGradientOperators<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 0, O, P>;
pub type StandardGradientOperatorsTransposed<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 0, P, O>;

pub trait FiniteElement<const G: usize, const M: usize, const N: usize> {
    fn integration_points() -> ParametricCoordinates<G, M>;
    fn integration_weights(&self) -> &ScalarList<G>;
    fn parametric_reference() -> ElementNodalReferenceCoordinates<N>;
    fn parametric_weights() -> ScalarList<G>;
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

pub struct Element<const G: usize, const N: usize, const O: usize> {
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: ScalarList<G>,
}

impl<const G: usize, const N: usize, const O: usize> Element<G, N, O> {
    fn gradient_vectors(&self) -> &GradientVectors<G, N> {
        &self.gradient_vectors
    }
}

impl<const G: usize, const N: usize, const O: usize> Debug for Element<G, N, O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N, O) {
            (1, 4, 1) => "LinearTetrahedron",
            (5, 5, 1) => "LinearPyramid",
            (6, 6, 1) => "LinearWedge",
            (8, 8, 1) => "LinearHexahedron",
            (4, 10, 0) => "CompositeTetrahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ G: {G}, N: {N} }}",)
    }
}

impl<const G: usize, const N: usize, const O: usize> Default for Element<G, N, O>
where
    Self: FiniteElement<G, 3, N> + From<ElementNodalReferenceCoordinates<N>>,
{
    fn default() -> Self {
        Self::parametric_reference().into()
    }
}

pub trait FiniteElementCreation<const G: usize, const N: usize>
where
    Self: Default + From<ElementNodalReferenceCoordinates<N>>,
{
}

impl<const G: usize, const N: usize, const O: usize> FiniteElementCreation<G, N>
    for Element<G, N, O>
where
    Self: Default + From<ElementNodalReferenceCoordinates<N>>,
{
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
