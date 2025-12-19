use crate::{
    constitutive::{ConstitutiveError, solid::elastic_hyperviscous::ElasticHyperviscous},
    fem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, FiniteElementError,
        solid::viscoelastic::ViscoelasticFiniteElement, surface::SurfaceElement,
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
        viscous_dissipation::<_, _, _, _, _, O>(
            self,
            constitutive_model,
            nodal_coordinates,
            nodal_velocities,
        )
    }
    fn dissipation_potential(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        dissipation_potential::<_, _, _, _, _, O>(
            self,
            constitutive_model,
            nodal_coordinates,
            nodal_velocities,
        )
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
        viscous_dissipation::<_, _, _, _, _, O>(
            self,
            constitutive_model,
            nodal_coordinates,
            nodal_velocities,
        )
    }
    fn dissipation_potential(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        dissipation_potential::<_, _, _, _, _, O>(
            self,
            constitutive_model,
            nodal_coordinates,
            nodal_velocities,
        )
    }
}

fn viscous_dissipation<C, F, const G: usize, const M: usize, const N: usize, const O: usize>(
    finite_element: &F,
    constitutive_model: &C,
    nodal_coordinates: &ElementNodalCoordinates<N>,
    nodal_velocities: &ElementNodalVelocities<N>,
) -> Result<Scalar, FiniteElementError>
where
    C: ElasticHyperviscous,
    F: ViscoelasticFiniteElement<C, G, M, N>,
{
    match finite_element
        .deformation_gradients(nodal_coordinates)
        .iter()
        .zip(
            finite_element
                .deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                .iter()
                .zip(finite_element.integration_weights()),
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
            format!("{finite_element:?}"),
        )),
    }
}

fn dissipation_potential<C, F, const G: usize, const M: usize, const N: usize, const O: usize>(
    finite_element: &F,
    constitutive_model: &C,
    nodal_coordinates: &ElementNodalCoordinates<N>,
    nodal_velocities: &ElementNodalVelocities<N>,
) -> Result<Scalar, FiniteElementError>
where
    C: ElasticHyperviscous,
    F: ViscoelasticFiniteElement<C, G, M, N>,
{
    match finite_element
        .deformation_gradients(nodal_coordinates)
        .iter()
        .zip(
            finite_element
                .deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                .iter()
                .zip(finite_element.integration_weights()),
        )
        .map(
            |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model
                        .dissipation_potential(deformation_gradient, deformation_gradient_rate)?
                        * integration_weight,
                )
            },
        )
        .sum()
    {
        Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
        Err(error) => Err(FiniteElementError::Upstream(
            format!("{error}"),
            format!("{finite_element:?}"),
        )),
    }
}
