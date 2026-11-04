use crate::{
    constitutive::solid::hyperelastic_plastic::HyperelasticPlastic,
    domain::block::solid::plastic::PlasticStateVariablesField,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError, solid::hyperelastic_plastic::HyperelasticPlasticFiniteElement,
            },
        },
        solid::{
            elastic_plastic::ElasticPlasticElements,
            hyperelastic_plastic::HyperelasticPlasticElements,
        },
    },
    math::Quantity,
    units::Energy,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H>
    HyperelasticPlasticElements<PlasticStateVariablesField<G>, 3> for Block<C, F, G, M, N, P, R, H>
where
    C: HyperelasticPlastic,
    F: HyperelasticPlasticFiniteElement<C, G, M, N, P>,
    Self: ElasticPlasticElements<PlasticStateVariablesField<G>, 3>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<G>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .map(|((element, nodes), state_variables_element)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    state_variables_element,
                )
            })
            .sum::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
