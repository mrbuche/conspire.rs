use crate::{
    constitutive::{ConstitutiveError, solid::hyperviscoelastic::Hyperviscoelastic},
    fem::block::element::{
        ElasticHyperviscousFiniteElement, Element, ElementNodalCoordinates, FiniteElementError,
        SolidFiniteElement,
    },
    math::{Scalar, Tensor},
};

pub trait HyperviscoelasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Hyperviscoelastic,
    Self: ElasticHyperviscousFiniteElement<C, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError>;
}

impl<C, const G: usize, const N: usize> HyperviscoelasticFiniteElement<C, G, N> for Element<G, N>
where
    C: Hyperviscoelastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(self.integration_weights().iter())
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
