pub mod internal_variables;

use crate::{
    constitutive::solid::elastic::Elastic,
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
    math::TensorArray,
};

/// TEMPORARY EXPERIMENT: a persistent pool with one thread per physical core.
fn pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("the thread pool")
    })
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H> ElasticElements<3>
    for Block<C, F, G, M, N, P, R, H>
where
    C: Elastic + Sync,
    F: ElasticFiniteElement<C, G, M, N, P> + Sync,
    R: Sync,
    H: Sync,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        use rayon::prelude::*;
        let mut scratch = self.scratch.forces.lock().expect("the scratch forces");
        scratch.resize_with(
            self.elements().len(),
            crate::mechanics::ForceList::<N>::zero,
        );
        let slots = scratch.as_mut_slice();
        pool()
            .install(|| {
                slots
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
        let gather = self
            .scratch
            .gather
            .get_or_init(|| super::super::Gather::new(self.connectivity().iter().as_slice()));
        let slots = &*slots;
        pool().install(|| {
            nodal_forces
                .as_mut_slice()
                .par_iter_mut()
                .with_min_len(512)
                .enumerate()
                .for_each(|(node, nodal_force)| {
                    if let Some(range) = gather.offsets.get(node..node + 2) {
                        gather.entries[range[0]..range[1]]
                            .iter()
                            .for_each(|&entry| *nodal_force += &slots[entry / N][entry % N])
                    }
                })
        });
        Ok(())
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
