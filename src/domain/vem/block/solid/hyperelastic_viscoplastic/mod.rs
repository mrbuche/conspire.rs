use crate::{
    constitutive::solid::hyperelastic_viscoplastic::HyperelasticViscoplastic,
    domain::{
        ElementModelError,
        block::solid::viscoplastic::ViscoplasticStateVariables,
        solid::{
            elastic_viscoplastic::ElasticViscoplasticElements,
            hyperelastic_viscoplastic::HyperelasticViscoplasticElements,
        },
    },
    math::{Differentiable, Quantity, Tensor},
    units::Energy,
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::{
                VirtualElementError,
                solid::hyperelastic_viscoplastic::HyperelasticViscoplasticVirtualElement,
            },
        },
    },
};

impl<C, F, Y> HyperelasticViscoplasticElements<ViscoplasticStateVariables<1, Y>, 3> for Block<C, F>
where
    C: HyperelasticViscoplastic<Y>,
    F: HyperelasticViscoplasticVirtualElement<C, Y>,
    Self: ElasticViscoplasticElements<ViscoplasticStateVariables<1, Y>, 3>,
    Y: Differentiable + Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<1, Y>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .zip(state_variables)
            .map(|((element, nodes), state_variables_element)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    state_variables_element,
                )
            })
            .sum::<Result<_, VirtualElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
