use crate::{
    constitutive::{ConstitutiveError, solid::hyperviscoelastic::Hyperviscoelastic},
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElementError,
        solid::elastic_hyperviscous::ElasticHyperviscousFiniteElement, surface::SurfaceElement,
    },
    math::{Scalar, Tensor},
};

pub trait HyperviscoelasticFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: Hyperviscoelastic,
    Self: ElasticHyperviscousFiniteElement<C, G, M, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    HyperviscoelasticFiniteElement<C, G, 3, N, P> for Element<G, N, O>
where
    C: Hyperviscoelastic,
    Self: ElasticHyperviscousFiniteElement<C, G, 3, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        helmholtz_free_energy::<_, _, _, _, _, O, _>(self, constitutive_model, nodal_coordinates)
    }
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    HyperviscoelasticFiniteElement<C, G, 2, N, P> for SurfaceElement<G, N, O>
where
    C: Hyperviscoelastic,
    Self: ElasticHyperviscousFiniteElement<C, G, 2, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
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
) -> Result<Scalar, FiniteElementError>
where
    C: Hyperviscoelastic,
    F: ElasticHyperviscousFiniteElement<C, G, M, N, P>,
{
    match element
        .deformation_gradients(nodal_coordinates)
        .iter()
        .zip(element.integration_weights())
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
            format!("{element:?}"),
        )),
    }
}
