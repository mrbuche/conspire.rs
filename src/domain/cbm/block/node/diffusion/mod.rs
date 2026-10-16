use super::Node;
use crate::{
    constitutive::{
        ConstitutiveError,
        diffusion::{Diffusion, DiffusionFlux, DiffusionGradient, DiffusionTangent, Diffusive},
    },
    domain::diffusion::NodalFields,
    math::{Quantity, Reference, TensorRank1},
    units::{ReciprocalLength, UnitMul},
};

pub type NodeStiffnesses<U> = Vec<Vec<Quantity<<U as Diffusive>::Stiffness>>>;

pub trait DiffusiveNode<U>
where
    U: Diffusive,
{
    fn gradient(&self, nodal_fields: &NodalFields<U>) -> TensorRank1<3, Reference, U::Gradient>;
}

impl<U> DiffusiveNode<U> for Node
where
    U: Diffusive,
    ReciprocalLength: UnitMul<U, Output = U::Gradient>,
{
    fn gradient(&self, nodal_fields: &NodalFields<U>) -> TensorRank1<3, Reference, U::Gradient> {
        self.neighbors()
            .iter()
            .zip(self.gradient_vectors())
            .map(|(&neighbor, gradient_vector)| gradient_vector * nodal_fields[neighbor])
            .sum()
    }
}

pub trait DiffusionNode<C>
where
    C: Diffusion,
    Self: DiffusiveNode<C::Field>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Field>,
    ) -> Result<Quantity<<C::Field as Diffusive>::Energy>, ConstitutiveError>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Field>,
    ) -> Result<Vec<Quantity<<C::Field as Diffusive>::Force>>, ConstitutiveError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Field>,
    ) -> Result<NodeStiffnesses<C::Field>, ConstitutiveError>;
}

impl<C> DiffusionNode<C> for Node
where
    C: Diffusion,
    Self: DiffusiveNode<C::Field>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Field>,
    ) -> Result<Quantity<<C::Field as Diffusive>::Energy>, ConstitutiveError> {
        let gradient: DiffusionGradient<C> = self.gradient(nodal_fields);
        Ok(constitutive_model.potential(&gradient)? * self.volume)
    }
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Field>,
    ) -> Result<Vec<Quantity<<C::Field as Diffusive>::Force>>, ConstitutiveError> {
        let gradient: DiffusionGradient<C> = self.gradient(nodal_fields);
        let flux: DiffusionFlux<C> = constitutive_model.flux(&gradient)?;
        Ok(self
            .gradient_vectors()
            .iter()
            .map(|gradient_vector| -(&flux * gradient_vector) * self.volume)
            .collect())
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Field>,
    ) -> Result<NodeStiffnesses<C::Field>, ConstitutiveError> {
        let gradient: DiffusionGradient<C> = self.gradient(nodal_fields);
        let tangent: DiffusionTangent<C> = constitutive_model.tangent(&gradient)?;
        Ok(self
            .gradient_vectors()
            .iter()
            .map(|gradient_vector_a| {
                self.gradient_vectors()
                    .iter()
                    .map(|gradient_vector_b| {
                        -(&(&tangent * gradient_vector_b) * gradient_vector_a) * self.volume
                    })
                    .collect()
            })
            .collect())
    }
}
