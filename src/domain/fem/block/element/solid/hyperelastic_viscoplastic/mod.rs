use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic_viscoplastic::HyperelasticViscoplastic},
    domain::block::element::solid::hyperelastic_viscoplastic::HyperelasticViscoplasticElement,
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            SolidElement, SolidFiniteElement,
            elastic_viscoplastic::ElasticViscoplasticFiniteElement,
            viscoplastic::ViscoplasticStateVariables,
        },
    },
    math::{Differentiable, Quantity, Tensor},
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
    Self: ElasticViscoplasticFiniteElement<C, G, M, N, P, Y>
        + HyperelasticViscoplasticElement<C, G, Y>,
    Y: Differentiable + Tensor,
{
}

impl<T, C, const G: usize, const M: usize, const N: usize, const P: usize, Y>
    HyperelasticViscoplasticFiniteElement<C, G, M, N, P, Y> for T
where
    C: HyperelasticViscoplastic<Y>,
    T: ElasticViscoplasticFiniteElement<C, G, M, N, P, Y>
        + HyperelasticViscoplasticElement<C, G, Y>,
    Y: Differentiable + Tensor,
{
}

impl<C, const G: usize, const N: usize, const O: usize, Y> HyperelasticViscoplasticElement<C, G, Y>
    for Element<3, G, N, O>
where
    C: HyperelasticViscoplastic<Y>,
    Self: SolidFiniteElement<G, 3, N, N>,
    Y: Differentiable + Tensor,
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
