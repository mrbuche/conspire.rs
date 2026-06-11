//! Finite element library.

pub mod block;
mod from;
pub mod solid;
pub mod thermal;

use crate::math::{
    TensorRank1Vec, TensorRank1Vec2D, TestError,
    optimize::{
        EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, OptimizationError,
        SecondOrderOptimization, ZerothOrderRootFinding,
    },
};
use std::fmt::{self, Debug, Display, Formatter};

pub type NodalCoordinates<const D: usize> = TensorRank1Vec<D, 1>;
pub type NodalCoordinatesHistory<const D: usize> = TensorRank1Vec2D<D, 1>;
pub type NodalReferenceCoordinates<const D: usize> = TensorRank1Vec<D, 0>;
pub type NodalVelocities<const D: usize> = TensorRank1Vec<D, 1>;
pub type NodalVelocitiesHistory<const D: usize> = TensorRank1Vec2D<D, 1>;

#[derive(Debug)]
pub struct Model<B, const D: usize> {
    blocks: B,
    coordinates: NodalReferenceCoordinates<D>,
}

#[derive(Debug)]
pub struct Blocks<B1, B2>(B1, B2);

#[derive(Debug)]
pub struct ElasticViscoplasticAndElastic<B1, B2>(B1, B2);

pub trait FiniteElementModel<const D: usize>
where
    Self: Debug,
{
    fn coordinates(&self) -> &NodalReferenceCoordinates<D>;
}

pub trait FiniteElements
where
    Self: Debug,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]);
}

impl<B, const D: usize> FiniteElements for Model<B, D>
where
    B: FiniteElements,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        self.blocks.node_neighbors(neighbors)
    }
}

impl<B1, B2> FiniteElements for Blocks<B1, B2>
where
    B1: FiniteElements,
    B2: FiniteElements,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        self.0.node_neighbors(neighbors);
        self.1.node_neighbors(neighbors)
    }
}

impl<B1, B2> FiniteElements for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: FiniteElements,
    B2: FiniteElements,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        self.0.node_neighbors(neighbors);
        self.1.node_neighbors(neighbors)
    }
}

impl<B, const D: usize> Model<B, D> {
    pub fn blocks(&self) -> &B {
        &self.blocks
    }
}

impl<B, const D: usize> FiniteElementModel<D> for Model<B, D>
where
    B: Debug,
{
    fn coordinates(&self) -> &NodalReferenceCoordinates<D> {
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

pub trait ZerothOrderRoot<X> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<X>,
    ) -> Result<X, OptimizationError>;
}

pub trait FirstOrderRoot<F, J, X> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<F, J, X>,
    ) -> Result<X, OptimizationError>;
}

pub trait FirstOrderMinimize<F, X> {
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<F, X>,
    ) -> Result<X, OptimizationError>;
}

pub trait SecondOrderMinimize<F, J, H, X> {
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<F, J, H, X>,
    ) -> Result<X, OptimizationError>;
}

impl<B, const D: usize> From<(B, NodalReferenceCoordinates<D>)> for Model<B, D> {
    fn from((blocks, coordinates): (B, NodalReferenceCoordinates<D>)) -> Self {
        Self {
            blocks,
            coordinates,
        }
    }
}

impl From<FiniteElementModelError> for TestError {
    fn from(error: FiniteElementModelError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
