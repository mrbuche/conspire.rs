pub mod internal_variables;

use crate::{
    constitutive::solid::elastic::Elastic,
    domain::solid::elastic::{ParallelElasticElements, Workers},
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError, planar::PlanarElasticFiniteElement,
                solid::elastic::ElasticFiniteElement,
            },
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticElements},
    },
    math::{Current, TensorArray, TensorRank1},
    mechanics::ForceList,
    units::Force,
};
use rayon::prelude::*;

/// TEMPORARY EXPERIMENT: the nodes that one task of the gather fills.
const GATHER_CHUNK: usize = 512;

/// The storage that a run of explicit dynamics reuses for the forces of a block.
pub struct ElementForces<const N: usize> {
    forces: Vec<ForceList<N>>,
    offsets: Vec<usize>,
    entries: Vec<usize>,
}

impl<const N: usize> ElementForces<N> {
    /// For each node, the flat indices `N * element + local` of its contributions, in element
    /// order, so that the gather adds in the order of the serial scatter.
    fn new(connectivity: &[[usize; N]]) -> Self {
        let nodes = connectivity
            .iter()
            .flatten()
            .max()
            .map_or(0, |&node| node + 1);
        let mut offsets = vec![0; nodes + 1];
        connectivity
            .iter()
            .flatten()
            .for_each(|&node| offsets[node + 1] += 1);
        (0..nodes).for_each(|node| offsets[node + 1] += offsets[node]);
        let mut next = offsets.clone();
        let mut entries = vec![0; connectivity.len() * N];
        connectivity
            .iter()
            .enumerate()
            .for_each(|(element, nodes)| {
                nodes.iter().enumerate().for_each(|(local, &node)| {
                    entries[next[node]] = N * element + local;
                    next[node] += 1;
                })
            });
        Self {
            forces: connectivity.iter().map(|_| ForceList::zero()).collect(),
            offsets,
            entries,
        }
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H>
    ParallelElasticElements<3> for Block<C, F, G, M, N, P, R, H>
where
    C: Elastic + Sync,
    F: ElasticFiniteElement<C, G, M, N, P> + Sync,
    R: Sync,
    H: Sync,
{
    type Scratch = ElementForces<N>;
    fn scratch(&self) -> ElementForces<N> {
        ElementForces::new(self.connectivity().iter().as_slice())
    }
    fn nodal_forces_parallel_into(
        &self,
        scratch: &mut ElementForces<N>,
        workers: &Workers,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        let ElementForces {
            forces,
            offsets,
            entries,
        } = scratch;
        workers
            .install(|| {
                forces
                    .par_iter_mut()
                    .zip(self.elements().par_iter())
                    .zip(self.connectivity().iter().as_slice().par_iter())
                    .try_for_each(|((slot, element), nodes)| {
                        *slot = element.nodal_forces(
                            self.constitutive_model(),
                            &Self::element_coordinates(nodal_coordinates, nodes),
                        )?;
                        Ok::<(), FiniteElementError>(())
                    })
            })
            .map_err(|error| ElementModelError::upstream(error, self))?;
        let (forces, offsets, entries) = (&*forces, &*offsets, &*entries);
        let nodes = offsets.len() - 1;
        workers.install(|| {
            nodal_forces.as_mut_slice()[..nodes]
                .par_chunks_mut(GATHER_CHUNK)
                .enumerate()
                .for_each(|(chunk, nodal_forces)| {
                    nodal_forces.iter_mut().enumerate().for_each(
                        |(index, nodal_force): (usize, &mut TensorRank1<3, Current, Force>)| {
                            let node = chunk * GATHER_CHUNK + index;
                            entries[offsets[node]..offsets[node + 1]]
                                .iter()
                                .for_each(|&entry| *nodal_force += &forces[entry / N][entry % N])
                        },
                    )
                })
        });
        Ok(())
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H> ElasticElements<3>
    for Block<C, F, G, M, N, P, R, H>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, M, N, P>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize, R> ElasticElements<2>
    for Block<C, F, G, 2, N, P, R>
where
    C: Elastic,
    F: PlanarElasticFiniteElement<C, G, N, P>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<2>,
        nodal_forces: &mut NodalForcesSolid<2>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<2>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<2>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
