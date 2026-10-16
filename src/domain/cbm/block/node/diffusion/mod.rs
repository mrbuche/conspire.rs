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

pub type NodeStiffnesses<L> = Vec<Vec<Quantity<<L as Diffusive>::Stiffness>>>;

pub trait DiffusiveNode<L>
where
    L: Diffusive,
{
    fn gradient(&self, nodal_fields: &NodalFields<L>) -> TensorRank1<3, Reference, L::Gradient>;
}

impl<L> DiffusiveNode<L> for Node
where
    L: Diffusive,
    ReciprocalLength: UnitMul<L::Field, Output = L::Gradient>,
{
    fn gradient(&self, nodal_fields: &NodalFields<L>) -> TensorRank1<3, Reference, L::Gradient> {
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
    Self: DiffusiveNode<C::Balance>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Balance>,
    ) -> Result<Quantity<<C::Balance as Diffusive>::Energy>, ConstitutiveError>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Balance>,
    ) -> Result<Vec<Quantity<C::Balance>>, ConstitutiveError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Balance>,
    ) -> Result<NodeStiffnesses<C::Balance>, ConstitutiveError>;
}

impl<C> DiffusionNode<C> for Node
where
    C: Diffusion,
    Self: DiffusiveNode<C::Balance>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Balance>,
    ) -> Result<Quantity<<C::Balance as Diffusive>::Energy>, ConstitutiveError> {
        let gradient: DiffusionGradient<C> = self.gradient(nodal_fields);
        Ok(constitutive_model.potential(&gradient)? * self.volume)
    }
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_fields: &NodalFields<C::Balance>,
    ) -> Result<Vec<Quantity<C::Balance>>, ConstitutiveError> {
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
        nodal_fields: &NodalFields<C::Balance>,
    ) -> Result<NodeStiffnesses<C::Balance>, ConstitutiveError> {
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
