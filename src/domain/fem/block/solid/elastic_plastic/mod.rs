use crate::{
    constitutive::solid::elastic_plastic::ElasticPlastic,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{FiniteElementError, solid::elastic_plastic::ElasticPlasticFiniteElement},
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic_plastic::ElasticPlasticElements},
    },
    math::{Quantity, TensorTupleListVec},
    mechanics::DeformationGradientPlastic,
};
use std::array::from_fn;

/// The rate-independent plastic state at every integration point of every element in a block.
pub type PlasticStateVariablesField<const G: usize> =
    TensorTupleListVec<DeformationGradientPlastic, Quantity, G>;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ElasticPlasticElements<PlasticStateVariablesField<G>, 3> for Block<C, F, G, M, N, P>
where
    C: ElasticPlastic,
    F: ElasticPlasticFiniteElement<C, G, M, N, P>,
{
    fn initial_state(&self) -> PlasticStateVariablesField<G> {
        self.elements()
            .iter()
            .map(|_| from_fn(|_| self.constitutive_model().initial_state()).into())
            .collect()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<G>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .try_for_each(|((element, nodes), state_variables_element)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        state_variables_element,
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
        state_variables: &PlasticStateVariablesField<G>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .try_for_each(|((element, nodes), state_variables_element)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        state_variables_element,
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
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<G>,
    ) -> Result<PlasticStateVariablesField<G>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .map(|((element, nodes), state_variables_element)| {
                element.updated_state(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    state_variables_element,
                )
            })
            .collect::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
