use crate::{
    constitutive::{
        ConstitutiveError,
        diffusion::{Diffusion, DiffusionFluxes, DiffusionTangents, Diffusive},
    },
    fem::block::element::{Element, FiniteElement, FiniteElementError},
    math::{Quantity, Reference, Tensor, TensorList, TensorRank1List},
    units::{ReciprocalLength, UnitMul},
};

pub type ElementNodalPotentials<U, const D: usize> = TensorList<Quantity<U>, D>;
pub type ElementNodalForces<U, const D: usize> = TensorList<Quantity<<U as Diffusive>::Force>, D>;
pub type ElementNodalStiffnesses<U, const D: usize> =
    TensorList<TensorList<Quantity<<U as Diffusive>::Stiffness>, D>, D>;
pub type ElementGradients<U, const G: usize> =
    TensorRank1List<3, Reference, G, <U as Diffusive>::Gradient>;

pub trait DiffusiveFiniteElement<U, const G: usize, const M: usize, const N: usize, const P: usize>
where
    U: Diffusive,
    Self: FiniteElement<G, M, N, P>,
{
    fn gradients(&self, nodal_potentials: &ElementNodalPotentials<U, N>) -> ElementGradients<U, G>;
}

impl<U, const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    DiffusiveFiniteElement<U, G, M, N, P> for Element<3, G, N, O>
where
    U: Diffusive,
    ReciprocalLength: UnitMul<U, Output = U::Gradient>,
    Self: FiniteElement<G, M, N, P>,
{
    fn gradients(&self, nodal_potentials: &ElementNodalPotentials<U, N>) -> ElementGradients<U, G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_potentials
                    .iter()
                    .zip(gradient_vectors)
                    .map(|(nodal_potential, gradient_vector)| gradient_vector * nodal_potential)
                    .sum()
            })
            .collect()
    }
}

pub trait DiffusionFiniteElement<C, const G: usize, const M: usize, const N: usize, const P: usize>
where
    C: Diffusion,
    Self: DiffusiveFiniteElement<C::Potential, G, M, N, P>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<C::Potential, N>,
    ) -> Result<Quantity<<C::Potential as Diffusive>::Energy>, FiniteElementError>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<C::Potential, N>,
    ) -> Result<ElementNodalForces<C::Potential, N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<C::Potential, N>,
    ) -> Result<ElementNodalStiffnesses<C::Potential, N>, FiniteElementError>;
}

impl<C, const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    DiffusionFiniteElement<C, G, M, N, P> for Element<3, G, N, O>
where
    C: Diffusion,
    Self: DiffusiveFiniteElement<C::Potential, G, M, N, P>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<C::Potential, N>,
    ) -> Result<Quantity<<C::Potential as Diffusive>::Energy>, FiniteElementError> {
        self.gradients(nodal_potentials)
            .iter()
            .zip(self.integration_weights())
            .map(|(gradient, integration_weight)| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model.potential(gradient)? * integration_weight,
                )
            })
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<C::Potential, N>,
    ) -> Result<ElementNodalForces<C::Potential, N>, FiniteElementError> {
        let fluxes = self
            .gradients(nodal_potentials)
            .iter()
            .map(|gradient| constitutive_model.flux(gradient))
            .collect::<Result<DiffusionFluxes<C, G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(fluxes
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(|(flux, (gradient_vectors, integration_weight))| {
                gradient_vectors
                    .iter()
                    .map(|gradient_vector| -(flux * gradient_vector) * integration_weight)
                    .collect()
            })
            .sum())
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<C::Potential, N>,
    ) -> Result<ElementNodalStiffnesses<C::Potential, N>, FiniteElementError> {
        let tangents = self
            .gradients(nodal_potentials)
            .iter()
            .map(|gradient| constitutive_model.tangent(gradient))
            .collect::<Result<DiffusionTangents<C, G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(tangents
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(|(tangent, (gradient_vectors, integration_weight))| {
                gradient_vectors
                    .iter()
                    .map(|gradient_vector_a| {
                        gradient_vectors
                            .iter()
                            .map(|gradient_vector_b| {
                                -(&(tangent * gradient_vector_b) * gradient_vector_a)
                                    * integration_weight
                            })
                            .collect()
                    })
                    .collect()
            })
            .sum())
    }
}
