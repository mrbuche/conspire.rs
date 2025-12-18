use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::Hyperelastic},
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElementError,
        solid::{SolidFiniteElement, elastic::ElasticFiniteElement},
    },
    math::{Scalar, Tensor},
};

pub trait HyperelasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Hyperelastic,
    Self: ElasticFiniteElement<C, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize> HyperelasticFiniteElement<C, G, N>
    for Element<G, N, O>
where
    C: Hyperelastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(self.integration_weights())
            .map(|(deformation_gradient, integration_weight)| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                        * integration_weight,
                )
            })
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
