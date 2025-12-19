use crate::{
    constitutive::{ConstitutiveError, solid::elastic_hyperviscous::ElasticHyperviscous},
    fem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, FiniteElement,
        FiniteElementError,
        solid::{SolidFiniteElement, viscoelastic::ViscoelasticFiniteElement},
        surface::SurfaceElement,
    },
    math::{Scalar, Tensor},
};

pub trait ElasticHyperviscousFiniteElement<C, const G: usize, const M: usize, const N: usize>
where
    C: ElasticHyperviscous,
    Self: ViscoelasticFiniteElement<C, G, M, N>,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError>;
    fn dissipation_potential(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize> ElasticHyperviscousFiniteElement<C, G, 3, N>
    for Element<G, N, O>
where
    C: ElasticHyperviscous,
    Self: ViscoelasticFiniteElement<C, G, 3, N>,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
                    Ok::<_, ConstitutiveError>(
                        constitutive_model
                            .viscous_dissipation(deformation_gradient, deformation_gradient_rate)?
                            * integration_weight,
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
    fn dissipation_potential(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
                    Ok::<_, ConstitutiveError>(
                        constitutive_model.dissipation_potential(
                            deformation_gradient,
                            deformation_gradient_rate,
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

impl<C, const G: usize, const N: usize, const O: usize> ElasticHyperviscousFiniteElement<C, G, 2, N>
    for SurfaceElement<G, N, O>
where
    C: ElasticHyperviscous,
    Self: ViscoelasticFiniteElement<C, G, 2, N>,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
                    Ok::<_, ConstitutiveError>(
                        constitutive_model
                            .viscous_dissipation(deformation_gradient, deformation_gradient_rate)?
                            * integration_weight,
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
    fn dissipation_potential(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
                    Ok::<_, ConstitutiveError>(
                        constitutive_model.dissipation_potential(
                            deformation_gradient,
                            deformation_gradient_rate,
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
