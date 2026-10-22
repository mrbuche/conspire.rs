#[cfg(test)]
mod test;

use crate::{
    constitutive::cohesive::{damage::Damage, elastic::Elastic},
    domain::{
        block::{add_node_neighbors, element::Elements, element_coordinates, trimmed_type_name},
        cohesive::damage::DamageElements,
        from::FromConnectivities,
    },
    fem::{
        ElementModelError, NodalCoordinates, NodalReferenceCoordinates,
        block::element::{
            ElementNodalReferenceCoordinates, FiniteElementError,
            cohesive::{
                damage::{DamageCohesiveElement, DamageState, initial_state},
                elastic::ElasticCohesiveElement,
            },
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticElements},
    },
    geometry::mesh::{Connectivity, PrimitiveConnectivity},
    math::integrate::StepError,
};
use std::fmt::{self, Debug, Formatter};

/// The greatest opening reached at every integration point of every element in a block.
pub type DamageStateField<const G: usize> = Vec<DamageState<G>>;

/// A block of cohesive elements, which join the two faces of an interface.
pub struct CohesiveBlock<C, F, const G: usize, const N: usize, const P: usize> {
    constitutive_model: C,
    connectivity: PrimitiveConnectivity<3, N>,
    elements: Vec<F>,
}

impl<C, F, const G: usize, const N: usize, const P: usize> CohesiveBlock<C, F, G, N, P> {
    fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn connectivity(&self) -> &PrimitiveConnectivity<3, N> {
        &self.connectivity
    }
    fn elements(&self) -> &[F] {
        &self.elements
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> Debug for CohesiveBlock<C, F, G, N, P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CohesiveBlock {{ constitutive model: {}, {} elements }}",
            trimmed_type_name::<C>(),
            self.elements().len()
        )
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> Elements
    for CohesiveBlock<C, F, G, N, P>
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(
            self.connectivity().iter().map(|nodes| nodes.as_slice()),
            neighbors,
        )
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> FromConnectivities<3, C>
    for CohesiveBlock<C, F, G, N, P>
where
    F: From<ElementNodalReferenceCoordinates<N>>,
    PrimitiveConnectivity<3, N>: TryFrom<Connectivity, Error = &'static str>,
{
    fn from_connectivities(
        mut connectivities: Vec<Connectivity>,
        constitutive_model: C,
        coordinates: &NodalReferenceCoordinates<3>,
    ) -> Result<Self, String> {
        if connectivities.len() != 1 {
            return Err(format!(
                "mesh has {} blocks, model type expects 1",
                connectivities.len()
            ));
        }
        let connectivity = PrimitiveConnectivity::<3, N>::try_from(connectivities.remove(0))?;
        let elements = connectivity
            .iter()
            .map(|nodes| element_coordinates(coordinates, nodes).into())
            .collect();
        Ok(Self {
            constitutive_model,
            connectivity,
            elements,
        })
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> ElasticElements<3>
    for CohesiveBlock<C, F, G, N, P>
where
    C: Elastic,
    F: ElasticCohesiveElement<C, G, N, P>,
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
                        &element_coordinates(nodal_coordinates, nodes),
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
                add_stiffnesses(
                    nodal_stiffnesses,
                    nodes,
                    element.nodal_stiffnesses(
                        self.constitutive_model(),
                        &element_coordinates(nodal_coordinates, nodes),
                    )?,
                );
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}

impl<C, F, const G: usize, const N: usize, const P: usize> DamageElements<DamageStateField<G>, 3>
    for CohesiveBlock<C, F, G, N, P>
where
    C: Damage,
    F: DamageCohesiveElement<C, G, N, P>,
{
    fn initial_state(&self) -> DamageStateField<G> {
        self.elements().iter().map(|_| initial_state()).collect()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state: &DamageStateField<G>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state)
            .try_for_each(|((element, nodes), state)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &element_coordinates(nodal_coordinates, nodes),
                        state,
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
        state: &DamageStateField<G>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state)
            .try_for_each(|((element, nodes), state)| {
                add_stiffnesses(
                    nodal_stiffnesses,
                    nodes,
                    element.nodal_stiffnesses(
                        self.constitutive_model(),
                        &element_coordinates(nodal_coordinates, nodes),
                        state,
                    )?,
                );
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state: &DamageStateField<G>,
    ) -> Result<DamageStateField<G>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state)
            .map(|((element, nodes), state)| {
                element.updated_state(
                    self.constitutive_model(),
                    &element_coordinates(nodal_coordinates, nodes),
                    state,
                )
            })
            .collect::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn step_error(
        &self,
        nodal_coordinates_old: &NodalCoordinates<3>,
        nodal_coordinates: &NodalCoordinates<3>,
        state: &DamageStateField<G>,
        updated_state: &DamageStateField<G>,
    ) -> Result<StepError, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state.iter().zip(updated_state))
            .try_fold(
                StepError::default(),
                |worst, ((element, nodes), (state_old, state))| {
                    element
                        .step_error(
                            self.constitutive_model(),
                            &element_coordinates(nodal_coordinates_old, nodes),
                            &element_coordinates(nodal_coordinates, nodes),
                            state_old,
                            state,
                        )
                        .map(|error| worst.max(error))
                },
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}

fn add_stiffnesses<const N: usize>(
    nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    nodes: &[usize; N],
    element_stiffnesses: crate::fem::block::element::solid::ElementNodalStiffnessesSolid<N>,
) {
    element_stiffnesses
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
}
