//! Finite element library.

pub mod block;
mod from;
pub mod solid;

use crate::{
    math::{
        TensorRank1Vec2D,
        optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizationError},
    },
    mechanics::Coordinates,
};
use std::fmt::{self, Debug, Display, Formatter};

pub type NodalCoordinates = Coordinates<1>;
pub type NodalCoordinatesHistory = TensorRank1Vec2D<3, 1>;
pub type NodalReferenceCoordinates = Coordinates<0>;
pub type NodalVelocities = Coordinates<1>;
pub type NodalVelocitiesHistory = TensorRank1Vec2D<3, 1>;

#[derive(Debug)]
pub struct Model<B> {
    blocks: B,
    coordinates: NodalReferenceCoordinates,
}

#[derive(Debug)]
pub struct Blocks<B1, B2>(B1, B2);

#[derive(Debug)]
pub struct ElasticViscoplasticAndElastic<B1, B2>(B1, B2);

pub trait FiniteElementModel
where
    Self: Debug,
{
    fn coordinates(&self) -> &NodalReferenceCoordinates;
}

impl<B> FiniteElementModel for Model<B>
where
    B: Debug,
{
    fn coordinates(&self) -> &NodalReferenceCoordinates {
        &self.coordinates
    }
}

pub enum FiniteElementModelError {
    Upstream(String, String),
}

impl Debug for FiniteElementModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, model) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In finite element model: {model}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

impl Display for FiniteElementModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, model) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In finite element model: {model}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

impl From<FiniteElementModelError> for String {
    fn from(error: FiniteElementModelError) -> Self {
        match error {
            FiniteElementModelError::Upstream(error, model) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In finite element model: {model}."
                )
            }
        }
    }
}

pub trait FirstOrderRoot<F, J, X> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<F, J, X>,
    ) -> Result<X, OptimizationError>;
}
