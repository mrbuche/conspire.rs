use crate::{
    constitutive::{
        ConstitutiveError,
        diffusion::{Diffusion, DiffusionFluxes, DiffusionTangents, Diffusive},
    },
    fem::block::element::{Element, FiniteElement, FiniteElementError},
    math::{Quantity, Reference, Tensor, TensorList, TensorRank1List},
    units::{ReciprocalLength, UnitMul},
};

pub type ElementNodalFields<L, const D: usize> = TensorList<Quantity<<L as Diffusive>::Field>, D>;
pub type ElementNodalForces<L, const D: usize> = TensorList<Quantity<L>, D>;
pub type ElementNodalStiffnesses<L, const D: usize> =
    TensorList<TensorList<Quantity<<L as Diffusive>::Stiffness>, D>, D>;
pub type ElementGradients<L, const G: usize> =
    TensorRank1List<3, Reference, G, <L as Diffusive>::Gradient>;

pub trait DiffusiveFiniteElement<L, const G: usize, const M: usize, const N: usize, const P: usize>
where
    L: Diffusive,
    Self: FiniteElement<G, M, N, P>,
{
    fn gradients(&self, nodal_fields: &ElementNodalFields<L, N>) -> ElementGradients<L, G>;
}

impl<L, const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    DiffusiveFiniteElement<L, G, M, N, P> for Element<3, G, N, O>
where
    L: Diffusive,
    ReciprocalLength: UnitMul<L::Field, Output = L::Gradient>,
    Self: FiniteElement<G, M, N, P>,
{
    fn gradients(&self, nodal_fields: &ElementNodalFields<L, N>) -> ElementGradients<L, G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_fields
                    .iter()
                    .zip(gradient_vectors)
                    .map(|(nodal_field, gradient_vector)| gradient_vector * nodal_field)
                    .sum()
            })
            .collect()
    }
}

pub trait DiffusionFiniteElement<C, const G: usize, const M: usize, const N: usize, const P: usize>
where
    C: Diffusion,
    Self: DiffusiveFiniteElement<C::Balance, G, M, N, P>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_fields: &ElementNodalFields<C::Balance, N>,
    ) -> Result<Quantity<<C::Balance as Diffusive>::Energy>, FiniteElementError>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_fields: &ElementNodalFields<C::Balance, N>,
    ) -> Result<ElementNodalForces<C::Balance, N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_fields: &ElementNodalFields<C::Balance, N>,
    ) -> Result<ElementNodalStiffnesses<C::Balance, N>, FiniteElementError>;
}

impl<C, const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    DiffusionFiniteElement<C, G, M, N, P> for Element<3, G, N, O>
where
    C: Diffusion,
    Self: DiffusiveFiniteElement<C::Balance, G, M, N, P>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_fields: &ElementNodalFields<C::Balance, N>,
    ) -> Result<Quantity<<C::Balance as Diffusive>::Energy>, FiniteElementError> {
        self.gradients(nodal_fields)
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
        nodal_fields: &ElementNodalFields<C::Balance, N>,
    ) -> Result<ElementNodalForces<C::Balance, N>, FiniteElementError> {
        let fluxes = self
            .gradients(nodal_fields)
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
        nodal_fields: &ElementNodalFields<C::Balance, N>,
    ) -> Result<ElementNodalStiffnesses<C::Balance, N>, FiniteElementError> {
        let tangents = self
            .gradients(nodal_fields)
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
