use crate::{
    constitutive::solid::hyperelastic_viscoplastic::HyperelasticViscoplastic,
    fem::{
        FiniteElementModelError, NodalCoordinates,
        block::{
            Block,
            element::solid::hyperelastic_viscoplastic::HyperelasticViscoplasticFiniteElement,
            solid::elastic_viscoplastic::ViscoplasticStateVariables,
        },
        solid::{
            elastic_viscoplastic::ElasticViscoplasticFiniteElements,
            hyperelastic_viscoplastic::HyperelasticViscoplasticFiniteElements,
        },
    },
    math::{Scalar, Tensor},
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, Y>
    HyperelasticViscoplasticFiniteElements<ViscoplasticStateVariables<G, Y>>
    for Block<C, F, G, M, N, P>
where
    C: HyperelasticViscoplastic<Y>,
    F: HyperelasticViscoplasticFiniteElement<C, G, M, N, P, Y>,
    Self: ElasticViscoplasticFiniteElements<ViscoplasticStateVariables<G, Y>>,
    Y: Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<Scalar, FiniteElementModelError> {
        match self
            .elements()
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
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
