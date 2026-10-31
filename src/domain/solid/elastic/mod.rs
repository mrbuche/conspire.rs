use crate::{
    domain::{
        Blocks, ElementModel, ElementModelError, Model, NodalCoordinates, NodalState,
        ProvidesTangent, Root, SolverFor,
        block::element::{Elements, Items},
        boundary::{Constrain, Load, LoadStiffness},
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Current, Tensor, TensorRank1, TensorRank2,
        optimize::{Direct, NewtonRaphson, OptimizationError, RootFinding},
    },
    units::{Force, ForcePerLength},
};
use rayon::prelude::*;
use std::ops::Index;

type NodalForce<const D: usize> = TensorRank1<D, Current, Force>;
type NodalStiffness<const D: usize> = TensorRank2<D, Current, Current, ForcePerLength>;

/// What a block says about its own items, from which its elastic assembly is derived.
///
/// An item is an element, or whatever else contributes forces to a set of nodes. Every
/// [`ElasticItems`] is [`ElasticElements`], with the forces and stiffnesses of the items added
/// to their nodes in the order of the items, and is [`ParallelElasticElements`] if it is `Sync`.
pub trait ElasticItems<const D: usize>
where
    Self: Items,
{
    /// The forces of an item, indexed by the position of a node among the nodes of the item.
    type Forces: Index<usize, Output = NodalForce<D>>;
    /// The stiffnesses of an item, indexed by the positions of two of its nodes.
    type Stiffnesses: Index<usize, Output: Index<usize, Output = NodalStiffness<D>>>;
    /// The forces of an item.
    fn forces_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Self::Forces, ElementModelError>;
    /// The stiffnesses of an item.
    fn stiffnesses_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Self::Stiffnesses, ElementModelError>;
}

impl<T, const D: usize> ElasticElements<D> for T
where
    T: ElasticItems<D>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        (0..self.number_of_items()).try_for_each(|item| {
            let forces = self.forces_of(item, nodal_coordinates)?;
            self.nodes_of(item)
                .iter()
                .enumerate()
                .for_each(|(local, &node)| nodal_forces[node] += &forces[local]);
            Ok(())
        })
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        (0..self.number_of_items()).try_for_each(|item| {
            let nodes = self.nodes_of(item);
            let stiffnesses = self.stiffnesses_of(item, nodal_coordinates)?;
            nodes.iter().enumerate().for_each(|(local_a, &node_a)| {
                nodes.iter().enumerate().for_each(|(local_b, &node_b)| {
                    nodal_stiffnesses[node_a][node_b] += &stiffnesses[local_a][local_b]
                })
            });
            Ok(())
        })
    }
}

/// The storage that a run reuses for the forces of the items of one block.
pub struct ItemScratch<F> {
    forces: Vec<Option<F>>,
    offsets: Vec<usize>,
    entries: Vec<(usize, usize)>,
}

impl<T, const D: usize> ParallelElasticElements<D> for T
where
    T: ElasticItems<D> + Sync,
    T::Forces: Send + Sync,
{
    type Scratch = ItemScratch<T::Forces>;
    fn scratch(&self) -> Self::Scratch {
        let items = self.number_of_items();
        let nodes = (0..items)
            .flat_map(|item| self.nodes_of(item))
            .max()
            .map_or(0, |&node| node + 1);
        let mut offsets = vec![0; nodes + 1];
        (0..items)
            .flat_map(|item| self.nodes_of(item))
            .for_each(|&node| offsets[node + 1] += 1);
        (0..nodes).for_each(|node| offsets[node + 1] += offsets[node]);
        let mut next = offsets.clone();
        let mut entries = vec![(0, 0); offsets[nodes]];
        (0..items).for_each(|item| {
            self.nodes_of(item)
                .iter()
                .enumerate()
                .for_each(|(local, &node)| {
                    entries[next[node]] = (item, local);
                    next[node] += 1;
                })
        });
        ItemScratch {
            forces: (0..items).map(|_| None).collect(),
            offsets,
            entries,
        }
    }
    fn nodal_forces_parallel_into(
        &self,
        scratch: &mut Self::Scratch,
        workers: &Workers,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        let ItemScratch {
            forces,
            offsets,
            entries,
        } = scratch;
        workers.install(|| {
            forces
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(item, slot)| {
                    *slot = Some(self.forces_of(item, nodal_coordinates)?);
                    Ok::<(), ElementModelError>(())
                })
        })?;
        let (forces, offsets, entries) = (&*forces, &*offsets, &*entries);
        let nodes = offsets.len() - 1;
        workers.install(|| {
            nodal_forces.as_mut_slice()[..nodes]
                .par_chunks_mut(GATHER_CHUNK)
                .enumerate()
                .for_each(|(chunk, nodal_forces)| {
                    nodal_forces
                        .iter_mut()
                        .enumerate()
                        .for_each(|(index, nodal_force)| {
                            let node = chunk * GATHER_CHUNK + index;
                            entries[offsets[node]..offsets[node + 1]].iter().for_each(
                                |&(item, local)| {
                                    *nodal_force +=
                                        &forces[item].as_ref().expect("the forces")[local]
                                },
                            )
                        })
                })
        });
        Ok(())
    }
}

/// The nodes that one task of the gather fills.
const GATHER_CHUNK: usize = 512;

pub trait ElasticElements<const D: usize>
where
    Self: Elements,
{
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
    B1: ElasticElements<D>,
    B2: ElasticElements<D>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.0.nodal_forces_into(nodal_coordinates, nodal_forces)?;
        self.1.nodal_forces_into(nodal_coordinates, nodal_forces)
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
        let sparse = S::SPARSE.then(|| self.sparse_solver(&equality_constraint, D, false));
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

/// TEMPORARY EXPERIMENT: the threads of one run of explicit dynamics.
///
/// This is the only type that names the thread pool, so replacing it replaces the backend.
#[cfg_attr(not(feature = "fem"), allow(dead_code))]
pub struct Workers(rayon::ThreadPool);

#[cfg_attr(not(feature = "fem"), allow(dead_code))]
impl Workers {
    /// Starts the given number of threads, which stop when the workers are dropped.
    pub fn new(threads: usize) -> Result<Self, String> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map(Self)
            .map_err(|error| error.to_string())
    }
    pub(crate) fn install<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.0.install(operation)
    }
}

/// Nodal forces that can be computed in parallel, with the storage of one run kept outside of
/// the model, so that the serial [`ElasticElements`] are untouched.
pub trait ParallelElasticElements<const D: usize>
where
    Self: ElasticElements<D>,
{
    /// The storage that a run reuses from one call to the next.
    type Scratch;
    /// Creates the storage for a run.
    fn scratch(&self) -> Self::Scratch;
    /// Adds the nodal forces to `nodal_forces`, in the same floating-point order as
    /// [`nodal_forces_into`](ElasticElements::nodal_forces_into).
    fn nodal_forces_parallel_into(
        &self,
        scratch: &mut Self::Scratch,
        workers: &Workers,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError>;
}

impl<B, const D: usize> ParallelElasticElements<D> for Model<B, D>
where
    B: ParallelElasticElements<D>,
{
    type Scratch = B::Scratch;
    fn scratch(&self) -> Self::Scratch {
        self.blocks.scratch()
    }
    fn nodal_forces_parallel_into(
        &self,
        scratch: &mut Self::Scratch,
        workers: &Workers,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_forces_parallel_into(scratch, workers, nodal_coordinates, nodal_forces)
    }
}

impl<B1, B2, const D: usize> ParallelElasticElements<D> for Blocks<B1, B2>
where
    B1: ParallelElasticElements<D>,
    B2: ParallelElasticElements<D>,
{
    type Scratch = (B1::Scratch, B2::Scratch);
    fn scratch(&self) -> Self::Scratch {
        (self.0.scratch(), self.1.scratch())
    }
    fn nodal_forces_parallel_into(
        &self,
        (first, second): &mut Self::Scratch,
        workers: &Workers,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_forces_parallel_into(first, workers, nodal_coordinates, nodal_forces)?;
        self.1
            .nodal_forces_parallel_into(second, workers, nodal_coordinates, nodal_forces)
    }
}
