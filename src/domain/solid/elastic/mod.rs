use crate::{
    domain::{
        Blocks, ElementModel, ElementModelError, Model, NodalCoordinates, NodalState,
        ProvidesTangent, Root, SolverFor,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        boundary::{Constrain, Load, LoadStiffness},
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Current, Tensor, TensorRank1,
        optimize::{Direct, NewtonRaphson, OptimizationError, RootFinding},
    },
    units::Force,
};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// TEMPORARY EXPERIMENT: when set, blocks are assembled one after another, each in parallel.
pub static SEQUENTIAL_BLOCKS: AtomicBool = AtomicBool::new(false);

/// TEMPORARY EXPERIMENT: a persistent pool with one thread per physical core.
pub(crate) fn pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(
                std::env::var("CONSPIRE_THREADS")
                    .ok()
                    .and_then(|threads| threads.parse().ok())
                    .unwrap_or(4),
            )
            .build()
            .expect("the thread pool")
    })
}

/// The nodes handed to one task of the gather.
const GATHER_CHUNK: usize = 512;

/// Computes the element forces of all blocks concurrently, then gathers them by node.
pub(crate) fn parallel_nodal_forces_into<B, const D: usize>(
    blocks: &B,
    nodal_coordinates: &NodalCoordinates<D>,
    nodal_forces: &mut NodalForcesSolid<D>,
) -> Result<(), ElementModelError>
where
    B: ElasticElements<D> + Sync,
{
    pool().install(|| blocks.compute_element_forces(nodal_coordinates))?;
    pool().install(|| {
        nodal_forces
            .as_mut_slice()
            .par_chunks_mut(GATHER_CHUNK)
            .enumerate()
            .for_each(|(chunk, forces)| blocks.add_node_forces(chunk * GATHER_CHUNK, forces))
    });
    Ok(())
}

pub trait ElasticElements<const D: usize>
where
    Self: Elements,
{
    /// TEMPORARY EXPERIMENT: whether the forces can be split into a parallel computation of the
    /// element forces and a gather of them by node.
    const PARALLEL: bool = false;
    /// Computes the element forces into storage kept by the block. Only called if `PARALLEL`.
    fn compute_element_forces(
        &self,
        _nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<(), ElementModelError> {
        Ok(())
    }
    /// Adds the forces on the nodes `first_node..first_node + forces.len()`, in element order.
    /// Only called if `PARALLEL`, after `compute_element_forces`.
    fn add_node_forces(&self, _first_node: usize, _forces: &mut [TensorRank1<D, Current, Force>]) {}
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        self.nodal_forces_into(nodal_coordinates, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        self.nodal_stiffnesses_into(nodal_coordinates, &mut nodal_stiffnesses)?;
        Ok(nodal_stiffnesses)
    }
}

impl<B, const D: usize> ElasticElements<D> for Model<B, D>
where
    B: ElasticElements<D>,
{
    const PARALLEL: bool = B::PARALLEL;
    fn compute_element_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks.compute_element_forces(nodal_coordinates)
    }
    fn add_node_forces(&self, first_node: usize, forces: &mut [TensorRank1<D, Current, Force>]) {
        self.blocks.add_node_forces(first_node, forces)
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_forces_into(nodal_coordinates, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_into(nodal_coordinates, nodal_stiffnesses)
    }
}

impl<B1, B2, const D: usize> ElasticElements<D> for Blocks<B1, B2>
where
    B1: ElasticElements<D> + Sync,
    B2: ElasticElements<D> + Sync,
{
    const PARALLEL: bool = B1::PARALLEL && B2::PARALLEL;
    fn compute_element_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<(), ElementModelError> {
        let (first, second) = rayon::join(
            || self.0.compute_element_forces(nodal_coordinates),
            || self.1.compute_element_forces(nodal_coordinates),
        );
        first?;
        second
    }
    fn add_node_forces(&self, first_node: usize, forces: &mut [TensorRank1<D, Current, Force>]) {
        self.0.add_node_forces(first_node, forces);
        self.1.add_node_forces(first_node, forces)
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        if Self::PARALLEL && !SEQUENTIAL_BLOCKS.load(Ordering::Relaxed) {
            parallel_nodal_forces_into(self, nodal_coordinates, nodal_forces)
        } else {
            self.0.nodal_forces_into(nodal_coordinates, nodal_forces)?;
            self.1.nodal_forces_into(nodal_coordinates, nodal_forces)
        }
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_into(nodal_coordinates, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_into(nodal_coordinates, nodal_stiffnesses)
    }
}

impl<B, const D: usize, S> SolverFor<Model<B, D>, NodalForcesSolid<D>, NodalStiffnessesSolid<D>>
    for NewtonRaphson<Direct, S>
where
    B: ElasticElements<D>,
{
    type Tangent = NodalStiffnessesSolid<D>;
    const SPARSE: bool = true;
}

impl<B, const D: usize> ProvidesTangent<NodalCoordinates<D>, NodalStiffnessesSolid<D>>
    for Model<B, D>
where
    B: ElasticElements<D>,
{
    fn provide_tangent(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        self.nodal_stiffnesses(nodal_coordinates)
    }
}

impl<B, const D: usize> Root<NodalForcesSolid<D>, NodalStiffnessesSolid<D>, NodalCoordinates<D>>
    for Model<B, D>
where
    B: ElasticElements<D>,
{
    fn root<S>(
        &self,
        boundary_conditions: &impl Constrain<<NodalCoordinates<D> as NodalState>::Layout>,
        applied_loads: &(
             impl Load<NodalCoordinates<D>, NodalForcesSolid<D>>
             + LoadStiffness<NodalCoordinates<D>, S::Tangent>
         ),
        solver: S,
    ) -> Result<NodalCoordinates<D>, OptimizationError>
    where
        S: SolverFor<Self, NodalForcesSolid<D>, NodalStiffnessesSolid<D>>
            + RootFinding<NodalForcesSolid<D>, S::Tangent, NodalCoordinates<D>>,
        Self: ProvidesTangent<NodalCoordinates<D>, S::Tangent>,
    {
        let equality_constraint = boundary_conditions.constraint();
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, D, false)
        });
        solver.root(
            |nodal_coordinates: &NodalCoordinates<D>| {
                let mut nodal_forces = self.nodal_forces(nodal_coordinates)?;
                applied_loads.subtract_from(nodal_coordinates, &mut nodal_forces)?;
                Ok(nodal_forces)
            },
            |nodal_coordinates: &NodalCoordinates<D>| {
                let mut tangent = self.provide_tangent(nodal_coordinates)?;
                applied_loads.add_stiffness_to(nodal_coordinates, &mut tangent)?;
                Ok(tangent)
            },
            self.coordinates().clone().into(),
            equality_constraint,
            sparse,
        )
    }
}
