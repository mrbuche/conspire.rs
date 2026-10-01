use crate::{
    constitutive::solid::elastic_plastic::ElasticPlastic,
    domain::{
        ElementModelError, block::solid::plastic::PlasticStateVariablesField,
        solid::elastic_plastic::ElasticPlasticElements,
    },
    math::optimize::NewtonRaphson,
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::{VirtualElementError, solid::elastic_plastic::ElasticPlasticVirtualElement},
            solid::{NodalForcesSolid, NodalStiffnessesSolid},
        },
    },
};
use std::array::from_fn;

impl<C, F> ElasticPlasticElements<PlasticStateVariablesField<1>, 3> for Block<C, F>
where
    C: ElasticPlastic,
    F: ElasticPlasticVirtualElement<C>,
{
    fn initial_state(&self) -> PlasticStateVariablesField<1> {
        self.elements()
            .iter()
            .map(|_| from_fn(|_| self.constitutive_model().initial_state()).into())
            .collect()
    }
    fn nodal_forces_and_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &PlasticStateVariablesField<1>,
        local_solver: &NewtonRaphson,
        nodal_forces: &mut NodalForcesSolid,
        nodal_stiffnesses: &mut NodalStiffnessesSolid,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .zip(state_variables)
            .try_for_each(|((element, nodes), state_variables_element)| {
                let (forces, stiffnesses) = element.nodal_forces_and_stiffnesses(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    state_variables_element,
                    local_solver,
                )?;
                forces
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                stiffnesses
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
                Ok::<(), VirtualElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &PlasticStateVariablesField<1>,
        local_solver: &NewtonRaphson,
    ) -> Result<PlasticStateVariablesField<1>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .zip(state_variables)
            .map(|((element, nodes), element_state_variables)| {
                element.updated_state(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    element_state_variables,
                    local_solver,
                )
            })
            .collect::<Result<_, VirtualElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
