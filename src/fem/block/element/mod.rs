#[cfg(test)]
mod test;

pub mod composite;
pub mod linear;
pub mod solid;
pub mod surface;
pub mod thermal;

use crate::{
    defeat_message,
    math::{Scalars, TensorRank1List, TensorRank1List2D, TestError},
    mechanics::{CurrentCoordinates, ReferenceCoordinates, Vectors2D},
};
use std::fmt::{self, Debug, Display, Formatter};

pub type ElementNodalCoordinates<const N: usize> = CurrentCoordinates<N>;
pub type ElementNodalVelocities<const N: usize> = CurrentCoordinates<N>;
pub type ElementNodalReferenceCoordinates<const N: usize> = ReferenceCoordinates<N>;
pub type GradientVectors<const G: usize, const N: usize> = Vectors2D<0, N, G>;
pub type ShapeFunctionsAtIntegrationPoints<const G: usize, const Q: usize> =
    TensorRank1List<Q, 9, G>;
pub type StandardGradientOperators<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 0, O, P>;
pub type StandardGradientOperatorsTransposed<const M: usize, const O: usize, const P: usize> =
    TensorRank1List2D<M, 0, P, O>;

pub struct Element<const G: usize, const N: usize> {
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars<G>,
}

impl<const G: usize, const N: usize> Element<G, N> {
    fn gradient_vectors(&self) -> &GradientVectors<G, N> {
        &self.gradient_vectors
    }
    fn integration_weights(&self) -> &Scalars<G> {
        &self.integration_weights
    }
}

impl<const G: usize, const N: usize> From<ElementNodalReferenceCoordinates<N>> for Element<G, N>
where
    Self: FiniteElement<G, N>,
{
    fn from(reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>) -> Self {
        let (gradient_vectors, integration_weights) = Self::initialize(reference_nodal_coordinates);
        Self {
            gradient_vectors,
            integration_weights,
        }
    }
}

impl<const G: usize, const N: usize> Debug for Element<G, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N) {
            (1, 4) => "LinearTetrahedron",
            (5, 5) => "LinearPyramid",
            (8, 8) => "LinearHexahedron",
            (4, 10) => "CompositeTetrahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ G: {G}, N: {N} }}",)
    }
}

pub trait FiniteElement<const G: usize, const N: usize>
where
    Self: From<ElementNodalReferenceCoordinates<N>>,
{
    fn initialize(
        reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>,
    ) -> (GradientVectors<G, N>, Scalars<G>);
    fn reset(&mut self);
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
