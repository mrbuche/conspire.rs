//! Finite element library.

pub mod block;
pub mod solid;

pub use crate::domain::{
    NodalCoordinates, NodalCoordinatesHistory, NodalForcesSolid, NodalReferenceCoordinates,
    NodalVelocities, NodalVelocitiesHistory,
};

use crate::math::optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizationError};
use std::fmt::{self, Debug, Display, Formatter};

// Consider using a model-to-block node map to avoid all the extra allocations/operations in nodal_forces etc.
// Would need to use a new trait for Blocks with the map in the receiver, and keep those maps in a Model field.
// Try to compare the performance before and after before making a decision.

// need to move solve routines from block to model
// would then need to change how unit tests work

// want to have mixed-C cases (viscoelastic + elastic) solve differently
// will have like (B1: Elastic, B2: Viscoelastic) => ViscoelasticFiniteElementModel
// just like constutituve/hybrid, and will have to impl combos specifically

#[derive(Debug)]
pub struct Model<B> {
    // blocks: B,
    // coordinates: NodalReferenceCoordinates,
    pub blocks: B,
    pub coordinates: NodalReferenceCoordinates,
    // pub is temporary until From<...> is implemented
}

#[derive(Debug)]
pub struct Blocks<B1, B2>(B1, B2);

pub struct Connectivities<C1, C2>(C1, C2);

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
