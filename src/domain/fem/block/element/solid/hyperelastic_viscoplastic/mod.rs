use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic_viscoplastic::HyperelasticViscoplastic},
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            SolidFiniteElement, elastic_viscoplastic::ElasticViscoplasticFiniteElement,
            viscoplastic::ViscoplasticStateVariables,
        },
    },
    math::{Differentiate, Quantity, Tensor},
    units::Energy,
};

pub trait HyperelasticViscoplasticFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
    Y,
> where
    C: HyperelasticViscoplastic<Y>,
    Self: ElasticViscoplasticFiniteElement<C, G, M, N, P, Y>,
    Y: Differentiate + Tensor,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<Quantity<Energy>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize, Y>
    HyperelasticViscoplasticFiniteElement<C, G, 3, N, P, Y> for Element<3, G, N, O>
where
    C: HyperelasticViscoplastic<Y>,
    Self: ElasticViscoplasticFiniteElement<C, G, 3, N, P, Y>,
    Y: Differentiate + Tensor,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<Quantity<Energy>, FiniteElementError> {
        self.deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .zip(self.integration_weights())
            .map(
                |((deformation_gradient, state_variable), integration_weight)| {
                    let (deformation_gradient_p, _) = state_variable.into();
                    Ok::<_, ConstitutiveError>(
                        constitutive_model.helmholtz_free_energy_density(
                            deformation_gradient,
                            deformation_gradient_p,
                        )? * integration_weight,
                    )
                },
            )
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
}
