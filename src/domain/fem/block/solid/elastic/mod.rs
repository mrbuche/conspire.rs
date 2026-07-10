use crate::{
    constitutive::solid::elastic::Elastic,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError, planar::PlanarElasticFiniteElement,
                solid::{ElementNodalStiffnessesSolid, elastic::ElasticFiniteElement},
            },
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticElements},
    },
};
use std::thread::{available_parallelism, scope};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> ElasticElements<3>
    for Block<C, F, G, M, N, P>
where
    C: Elastic + Sync,
    F: ElasticFiniteElement<C, G, M, N, P> + Sync,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        match self
            .elements()
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
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        let mut results: Vec<Option<Result<ElementNodalStiffnessesSolid<N>, FiniteElementError>>> =
            (0..elements.len()).map(|_| None).collect();
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let chunk_size = elements.len().div_ceil(threads).max(1);
        scope(|scope| {
            let constitutive_model = self.constitutive_model();
            elements
                .chunks(chunk_size)
                .zip(connectivity.chunks(chunk_size))
                .zip(results.chunks_mut(chunk_size))
                .for_each(|((elements_chunk, nodes_chunk), results_chunk)| {
                    scope.spawn(move || {
                        elements_chunk
                            .iter()
                            .zip(nodes_chunk.iter())
                            .zip(results_chunk.iter_mut())
                            .for_each(|((element, &nodes), result)| {
                                *result = Some(element.nodal_stiffnesses(
                                    constitutive_model,
                                    &Self::element_coordinates(nodal_coordinates, nodes),
                                ));
                            });
                    });
                });
        });
        match results
            .into_iter()
            .zip(self.connectivity())
            .try_for_each(|(result, nodes)| {
                result
                    .expect("every element should have been assigned a result")?
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
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> ElasticElements<2>
    for Block<C, F, G, 2, N, P>
where
    C: Elastic,
    F: PlanarElasticFiniteElement<C, G, N, P>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<2>,
        nodal_forces: &mut NodalForcesSolid<2>,
    ) -> Result<(), ElementModelError> {
        match self
            .elements()
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
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<2>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<2>,
    ) -> Result<(), ElementModelError> {
        match self
            .elements()
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
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
