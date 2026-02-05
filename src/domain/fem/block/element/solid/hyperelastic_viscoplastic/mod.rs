use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic_viscoplastic::HyperelasticViscoplastic},
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            SolidFiniteElement, elastic_viscoplastic::ElasticViscoplasticFiniteElement,
            viscoplastic::ViscoplasticStateVariables,
        },
    },
    math::{Scalar, Tensor},
};

pub trait HyperelasticViscoplasticFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: HyperelasticViscoplastic,
    Self: ElasticViscoplasticFiniteElement<C, G, M, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<Scalar, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    HyperelasticViscoplasticFiniteElement<C, G, 3, N, P> for Element<G, N, O>
where
    C: HyperelasticViscoplastic,
    Self: ElasticViscoplasticFiniteElement<C, G, 3, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
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
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
