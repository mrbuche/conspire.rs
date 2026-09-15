//! Finite element methods.

#[cfg(test)]
mod test;

pub mod block;
mod from;
pub mod solid;
pub mod thermal;

use crate::math::{
    Style, StyledError,
    assert::AssertionError,
    optimize::{
        EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, OptimizationError,
        SecondOrderOptimization, ZerothOrderRootFinding,
    },
    styled_error,
};
use std::fmt::{Debug, Display};

pub(crate) use crate::domain::nodal_coordinates;
pub use crate::domain::{
    NodalCoordinates, NodalCoordinatesHistory, NodalReferenceCoordinates, NodalVelocities,
    NodalVelocitiesHistory,
};

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

pub use crate::domain::block::element::Elements;

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

impl ElementModelError {
    pub fn upstream(error: impl Display, context: &(impl Debug + ?Sized)) -> Self {
        Self::Upstream(format!("{error}"), format!("{context:?}"))
    }
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

pub trait ZerothOrderRoot<F, X> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<F, X>,
    ) -> Result<X, OptimizationError>;
}

pub trait FirstOrderRoot<F, J, X> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<F, J, X>,
    ) -> Result<X, OptimizationError>;
}

pub trait FirstOrderMinimize<F, J, X> {
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<F, J, X>,
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
