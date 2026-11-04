use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic_plastic::HyperelasticPlastic},
    domain::block::element::solid::hyperelastic_plastic::HyperelasticPlasticElement,
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            SolidElement, SolidFiniteElement, elastic_plastic::ElasticPlasticFiniteElement,
            plastic::PlasticStateVariables,
        },
    },
    math::{Quantity, Tensor},
    units::Energy,
};

pub trait HyperelasticPlasticFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: HyperelasticPlastic,
    Self: ElasticPlasticFiniteElement<C, G, M, N, P> + HyperelasticPlasticElement<C, G>,
{
}

impl<T, C, const G: usize, const M: usize, const N: usize, const P: usize>
    HyperelasticPlasticFiniteElement<C, G, M, N, P> for T
where
    C: HyperelasticPlastic,
    T: ElasticPlasticFiniteElement<C, G, M, N, P> + HyperelasticPlasticElement<C, G>,
{
}

impl<C, const G: usize, const N: usize, const O: usize> HyperelasticPlasticElement<C, G>
    for Element<3, G, N, O>
where
    C: HyperelasticPlastic,
    Self: SolidFiniteElement<G, 3, N, N>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<Quantity<Energy>, FiniteElementError> {
        self.deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .zip(self.integration_weights())
            .map(
                |((deformation_gradient, state_variable), integration_weight)| {
                    Ok(constitutive_model
                        .helmholtz_free_energy_density(deformation_gradient, &state_variable.0)?
                        * integration_weight)
                },
            )
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
}
