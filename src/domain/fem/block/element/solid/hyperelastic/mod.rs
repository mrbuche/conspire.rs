#[cfg(feature = "autodiff")]
pub mod autodiff;
pub mod internal_variables;

use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::Hyperelastic},
    domain::block::element::solid::hyperelastic::HyperelasticElement,
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElementError, solid::elastic::ElasticFiniteElement,
        surface::SurfaceElement,
    },
    math::{Quantity, Tensor},
    units::Energy,
};

pub trait HyperelasticFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: Hyperelastic,
    Self: ElasticFiniteElement<C, G, M, N, P> + HyperelasticElement<C>,
{
}

impl<T, C, const G: usize, const M: usize, const N: usize, const P: usize>
    HyperelasticFiniteElement<C, G, M, N, P> for T
where
    C: Hyperelastic,
    T: ElasticFiniteElement<C, G, M, N, P> + HyperelasticElement<C>,
{
}

impl<C, const G: usize, const N: usize, const O: usize> HyperelasticElement<C>
    for Element<3, G, N, O>
where
    C: Hyperelastic,
    Self: ElasticFiniteElement<C, G, 3, N, N>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Quantity<Energy>, FiniteElementError> {
        helmholtz_free_energy::<_, _, _, _, _, O, _>(self, constitutive_model, nodal_coordinates)
    }
}

impl<C, const G: usize, const N: usize, const O: usize> HyperelasticElement<C>
    for SurfaceElement<G, N, O>
where
    C: Hyperelastic,
    Self: ElasticFiniteElement<C, G, 2, N, N>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Quantity<Energy>, FiniteElementError> {
        helmholtz_free_energy::<_, _, _, _, _, O, _>(self, constitutive_model, nodal_coordinates)
    }
}

fn helmholtz_free_energy<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
>(
    element: &F,
    constitutive_model: &C,
    nodal_coordinates: &ElementNodalCoordinates<N>,
) -> Result<Quantity<Energy>, FiniteElementError>
where
    C: Hyperelastic,
    F: ElasticFiniteElement<C, G, M, N, P>,
{
    element
        .deformation_gradients(nodal_coordinates)
        .iter()
        .zip(element.integration_weights())
        .map(|(deformation_gradient, integration_weight)| {
            Ok::<_, ConstitutiveError>(
                constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                    * integration_weight,
            )
        })
        .sum::<Result<_, ConstitutiveError>>()
        .map_err(|error| FiniteElementError::upstream(error, element))
}
