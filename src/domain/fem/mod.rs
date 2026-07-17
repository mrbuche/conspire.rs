//! Finite element library.

pub mod block;
mod from;
pub mod solid;
pub mod thermal;

use crate::math::{
    Style, StyledError, TensorRank1Vec, TensorRank1Vec2D,
    assert::AssertionError,
    optimize::{
        EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, OptimizationError,
        SecondOrderOptimization, ZerothOrderRootFinding,
    },
    styled_error,
};
use std::fmt::Debug;

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

pub trait ElementModel<const D: usize>
where
    Self: Debug,
{
    fn coordinates(&self) -> &NodalReferenceCoordinates<D>;
}

pub trait Elements
where
    Self: Debug,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]);
}

impl<B, const D: usize> Elements for Model<B, D>
where
    B: Elements,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        self.blocks.node_neighbors(neighbors)
    }
}

impl<B1, B2> Elements for Blocks<B1, B2>
where
    B1: Elements,
    B2: Elements,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        self.0.node_neighbors(neighbors);
        self.1.node_neighbors(neighbors)
    }
}

impl<B1, B2> Elements for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: Elements,
    B2: Elements,
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

impl<B, const D: usize> ElementModel<D> for Model<B, D>
where
    B: Debug,
{
    fn coordinates(&self) -> &NodalReferenceCoordinates<D> {
        &self.coordinates
    }
}

pub enum ElementModelError {
    Upstream(String, String),
}

impl From<ElementModelError> for String {
    fn from(error: ElementModelError) -> Self {
        error.message(&Style::detect())
    }
}

impl StyledError for ElementModelError {
    fn message(&self, style: &Style) -> String {
        let c = style.frame;
        match self {
            Self::Upstream(error, model) => format!(
                "{error}{c}\n\
                In element model: {model}."
            ),
        }
    }
}

styled_error!(ElementModelError);

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

impl From<ElementModelError> for AssertionError {
    fn from(error: ElementModelError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
