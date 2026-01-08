use crate::{
    constitutive::{ConstitutiveError, solid::elastic_hyperviscous::ElasticHyperviscous},
    fem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, FiniteElementError,
        solid::viscoelastic::ViscoelasticFiniteElement, surface::SurfaceElement,
    },
    math::{Scalar, Tensor},
};

pub trait ElasticHyperviscousFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: ElasticHyperviscous,
    Self: ViscoelasticFiniteElement<C, G, M, N, P>,
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

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    ElasticHyperviscousFiniteElement<C, G, 3, N, P> for Element<G, N, O>
where
    C: ElasticHyperviscous,
    Self: ViscoelasticFiniteElement<C, G, 3, N, P>,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        viscous_dissipation::<_, _, _, _, _, O, _>(
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
        dissipation_potential::<_, _, _, _, _, O, _>(
            self,
            constitutive_model,
            nodal_coordinates,
            nodal_velocities,
        )
    }
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    ElasticHyperviscousFiniteElement<C, G, 2, N, P> for SurfaceElement<G, N, O>
where
    C: ElasticHyperviscous,
    Self: ViscoelasticFiniteElement<C, G, 2, N, P>,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        viscous_dissipation::<_, _, _, _, _, O, _>(
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
        dissipation_potential::<_, _, _, _, _, O, _>(
            self,
            constitutive_model,
            nodal_coordinates,
            nodal_velocities,
        )
    }
}

fn viscous_dissipation<
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
    nodal_velocities: &ElementNodalVelocities<N>,
) -> Result<Scalar, FiniteElementError>
where
    C: ElasticHyperviscous,
    F: ViscoelasticFiniteElement<C, G, M, N, P>,
{
    match element
        .deformation_gradients(nodal_coordinates)
        .iter()
        .zip(
            element
                .deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                .iter()
                .zip(element.integration_weights()),
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
            format!("{element:?}"),
        )),
    }
}

fn dissipation_potential<
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
    nodal_velocities: &ElementNodalVelocities<N>,
) -> Result<Scalar, FiniteElementError>
where
    C: ElasticHyperviscous,
    F: ViscoelasticFiniteElement<C, G, M, N, P>,
{
    match element
        .deformation_gradients(nodal_coordinates)
        .iter()
        .zip(
            element
                .deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                .iter()
                .zip(element.integration_weights()),
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
            format!("{element:?}"),
        )),
    }
}
