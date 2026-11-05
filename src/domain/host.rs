//! What the host needs of a domain, on top of the elements: the model, with its blocks.

use super::NodalReferenceCoordinates;
pub(crate) use crate::{
    domain::{
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        boundary::{Conservative, Constrain, Load, LoadStiffness, NodalState},
    },
    geometry::{
        Coordinates,
        mesh::{NodeSets, SideSets},
    },
    math::{
        Style, StyledError, Tensor,
        assert::AssertionError,
        optimize::{EqualityConstraint, Optimization, OptimizationError, RootFinding},
        sparse::SparseSolver,
        styled_error,
    },
};
pub(crate) use core::fmt::{Debug, Display};

pub(crate) fn nodal_coordinates<const D: usize>(
    coordinates: Coordinates<D>,
) -> NodalReferenceCoordinates<D> {
    coordinates
        .into_iter()
        .map(|coordinate| coordinate.with_unit())
        .collect()
}

#[derive(Debug)]
pub struct Model<B, const D: usize> {
    pub(crate) blocks: B,
    pub(crate) coordinates: NodalReferenceCoordinates<D>,
    pub(crate) node_sets: NodeSets,
    pub(crate) side_sets: SideSets,
}

impl<B, const D: usize> Model<B, D> {
    /// The sets of nodes the model took from its mesh.
    pub fn node_sets(&self) -> &NodeSets {
        &self.node_sets
    }
    /// The sets of element sides the model took from its mesh.
    pub fn side_sets(&self) -> &SideSets {
        &self.side_sets
    }
}

#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
#[derive(Debug)]
pub struct Blocks<B1, B2>(pub B1, pub B2);

#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
#[derive(Debug)]
pub struct ElasticViscoplasticAndElastic<B1, B2>(pub(crate) B1, pub(crate) B2);

pub trait ElementModel<const D: usize>
where
    Self: Debug,
{
    fn coordinates(&self) -> &NodalReferenceCoordinates<D>;
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

#[cfg_attr(
    not(any(feature = "fem", feature = "vem", feature = "cbm")),
    allow(dead_code)
)]
impl<B, const D: usize> Model<B, D>
where
    B: Elements,
{
    /// The nodes that share an element with each node, sorted and without repeats.
    pub(crate) fn finalized_neighbors(&self) -> Vec<Vec<usize>> {
        let mut neighbors = vec![Vec::new(); self.coordinates.len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        neighbors
    }
    /// The sparse solver for the nodes that share an element, with `dimension` unknowns for each
    /// node.
    pub(crate) fn sparse_solver(
        &self,
        equality_constraints: &[EqualityConstraint],
        dimension: usize,
        symmetric: bool,
    ) -> SparseSolver {
        solver_from_neighbors(
            &self.finalized_neighbors(),
            equality_constraints,
            dimension,
            symmetric,
        )
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

#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub trait Root<F, J, X>
where
    Self: Sized,
    X: NodalState,
{
    fn root<S>(
        &self,
        boundary_conditions: &impl Constrain<X::Layout>,
        applied_loads: &(impl Load<X, F> + LoadStiffness<X, S::Tangent>),
        solver: S,
    ) -> Result<X, OptimizationError>
    where
        S: SolverFor<Self, F, J> + RootFinding<F, S::Tangent, X>,
        Self: ProvidesTangent<X, S::Tangent>;
}

#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub trait Minimize<F, J, X>
where
    Self: Sized,
    X: NodalState,
{
    fn minimize<S>(
        &self,
        boundary_conditions: &impl Constrain<X::Layout>,
        applied_loads: &(impl Load<X, J> + Conservative<X, F>),
        solver: S,
    ) -> Result<X, OptimizationError>
    where
        S: SolverFor<Self, F, J> + Optimization<F, J, S::Tangent, X>,
        Self: ProvidesTangent<X, S::Tangent>;
}

impl<B, const D: usize> From<(B, NodalReferenceCoordinates<D>)> for Model<B, D> {
    fn from((blocks, coordinates): (B, NodalReferenceCoordinates<D>)) -> Self {
        Self {
            blocks,
            coordinates,
            node_sets: NodeSets::new(),
            side_sets: SideSets::new(),
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

/// A solver that can minimize a model, and the tangent it works from.
///
/// The family of the model, being carried by the energy and force types,
/// keeps the implementations for one solver from overlapping.
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub trait SolverFor<M, F, J> {
    type Tangent;
    /// Whether the solver wants the sparse structure of the assembled tangent.
    const SPARSE: bool;
}

/// A model that can hand out its tangent in the form a solver asks for.
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub trait ProvidesTangent<X, T> {
    fn provide_tangent(&self, argument: &X) -> Result<T, ElementModelError>;
}
