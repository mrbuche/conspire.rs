pub mod internal_variables;

use rayon::prelude::*;

use crate::{
    constitutive::solid::elastic::Elastic,
    domain::solid::elastic::{parallel_nodal_forces_into, pool},
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
    units::Force,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H> ElasticElements<3>
    for Block<C, F, G, M, N, P, R, H>
where
    C: Elastic + Sync,
    F: ElasticFiniteElement<C, G, M, N, P> + Sync,
    R: Sync,
    H: Sync,
{
    const PARALLEL: bool = true;
    fn compute_element_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<(), ElementModelError> {
        let mut scratch = self.scratch.forces.write().expect("the scratch forces");
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
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn add_node_forces(&self, first_node: usize, forces: &mut [TensorRank1<3, Current, Force>]) {
        let gather = self
            .scratch
            .gather
            .get_or_init(|| super::super::Gather::new(self.connectivity().iter().as_slice()));
        let slots = self.scratch.forces.read().expect("the scratch forces");
        forces.iter_mut().enumerate().for_each(|(index, force)| {
            if let Some(range) = gather
                .offsets
                .get(first_node + index..first_node + index + 2)
            {
                gather.entries[range[0]..range[1]]
                    .iter()
                    .for_each(|&entry| *force += &slots[entry / N][entry % N])
            }
        })
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        parallel_nodal_forces_into(self, nodal_coordinates, nodal_forces)
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
