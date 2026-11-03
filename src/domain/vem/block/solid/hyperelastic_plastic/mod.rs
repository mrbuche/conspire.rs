use crate::{
    constitutive::solid::hyperelastic_plastic::HyperelasticPlastic,
    domain::{
        ElementModelError,
        block::solid::plastic::PlasticStateVariablesField,
        solid::{
            elastic_plastic::ElasticPlasticElements,
            hyperelastic_plastic::HyperelasticPlasticElements,
        },
    },
    math::Quantity,
    units::Energy,
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::{
                VirtualElementError, solid::hyperelastic_plastic::HyperelasticPlasticVirtualElement,
            },
        },
    },
};

impl<C, F, R> HyperelasticPlasticElements<PlasticStateVariablesField<1>, 3> for Block<C, F, R>
where
    C: HyperelasticPlastic,
    F: HyperelasticPlasticVirtualElement<C>,
    Self: ElasticPlasticElements<PlasticStateVariablesField<1>, 3>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &PlasticStateVariablesField<1>,
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
