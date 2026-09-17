use crate::{
    constitutive::solid::elastic_viscoplastic::ElasticViscoplastic,
    domain::{
        ElementModelError,
        block::solid::viscoplastic::{ViscoplasticEvolution, ViscoplasticStateVariables},
        solid::elastic_viscoplastic::ElasticViscoplasticElements,
    },
    math::{Differentiable, Tensor},
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::{
                VirtualElementError, solid::elastic_viscoplastic::ElasticViscoplasticVirtualElement,
            },
            solid::{NodalForcesSolid, NodalStiffnessesSolid},
        },
    },
};
use std::array::from_fn;

impl<C, F, Y> ElasticViscoplasticElements<ViscoplasticStateVariables<1, Y>, 3> for Block<C, F>
where
    C: ElasticViscoplastic<Y>,
    F: ElasticViscoplasticVirtualElement<C, Y>,
    Y: Differentiable + Tensor,
{
    fn initial_state(&self) -> ViscoplasticStateVariables<1, Y> {
        self.elements()
            .iter()
            .map(|_| from_fn(|_| self.constitutive_model().initial_state()).into())
            .collect()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<1, Y>,
        nodal_forces: &mut NodalForcesSolid,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
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
                Ok::<(), VirtualElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<1, Y>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
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
                Ok::<(), VirtualElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<1, Y>,
    ) -> Result<ViscoplasticEvolution<1, Y>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .zip(state_variables)
            .map(|((element, nodes), element_state_variables)| {
                element.state_variables_evolution(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    element_state_variables,
                )
            })
            .collect::<Result<_, VirtualElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
