pub mod internal_variables;

use crate::{
    constitutive::solid::elastic::Elastic,
    domain::solid::elastic::ElasticItems,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError,
                planar::PlanarElasticFiniteElement,
                solid::{
                    ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
                    elastic::ElasticFiniteElement,
                },
            },
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticElements},
    },
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H> ElasticItems<3>
    for Block<C, F, G, M, N, P, R, H>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, M, N, P>,
{
    type Forces = ElementNodalForcesSolid<N>;
    type Stiffnesses = ElementNodalStiffnessesSolid<N>;
    fn forces_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementNodalForcesSolid<N>, ElementModelError> {
        self.elements()[item]
            .nodal_forces(
                self.constitutive_model(),
                &Self::element_coordinates(
                    nodal_coordinates,
                    &self.connectivity().iter().as_slice()[item],
                ),
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn stiffnesses_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, ElementModelError> {
        self.elements()[item]
            .nodal_stiffnesses(
                self.constitutive_model(),
                &Self::element_coordinates(
                    nodal_coordinates,
                    &self.connectivity().iter().as_slice()[item],
                ),
            )
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
