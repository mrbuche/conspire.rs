pub mod solid;

use crate::{
    defeat_message,
    math::{Scalars, TensorRank1, TensorRank1Vec2D, TestError},
    mechanics::Vectors2D,
};
use std::fmt::{self, Debug, Display, Formatter};

pub type ElementNodalCoordinates<'a> = Vec<&'a TensorRank1<3, 1>>;
pub type ElementNodalReferenceCoordinates = TensorRank1Vec2D<3, 0>;
pub type GradientVectors = Vectors2D<0>;

pub struct Element {
    gradient_vectors: GradientVectors,
    integration_weights: Scalars,
}

impl Element {
    fn gradient_vectors(&self) -> &GradientVectors {
        &self.gradient_vectors
    }
    fn integration_weights(&self) -> &Scalars {
        &self.integration_weights
    }
}

impl Debug for Element {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "VirtualElement {{ ... }}",)
    }
}

pub trait VirtualElement
where
    Self: From<ElementNodalReferenceCoordinates>,
{
    fn initialize(
        reference_nodal_coordinates: ElementNodalReferenceCoordinates,
    ) -> (GradientVectors, Scalars);
}

pub enum VirtualElementError {
    Upstream(String, String),
}

impl From<VirtualElementError> for TestError {
    fn from(error: VirtualElementError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl Debug for VirtualElementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, element) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In virtual element: {element}."
                )
            }
        };
        write!(f, "\n{error}\n\x1b[0;2;31m{}\x1b[0m\n", defeat_message())
    }
}

impl Display for VirtualElementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, element) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In virtual element: {element}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}
