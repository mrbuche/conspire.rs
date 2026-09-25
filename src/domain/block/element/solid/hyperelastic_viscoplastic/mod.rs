use crate::{
    constitutive::solid::hyperelastic_viscoplastic::HyperelasticViscoplastic,
    domain::block::element::solid::{
        elastic_viscoplastic::ElasticViscoplasticElement, viscoplastic::ViscoplasticStateVariables,
    },
    math::{Differentiable, Quantity, Tensor},
    units::Energy,
};

pub trait HyperelasticViscoplasticElement<C, const G: usize, Y>
where
    C: HyperelasticViscoplastic<Y>,
    Self: ElasticViscoplasticElement<C, G, Y>,
    Y: Differentiable + Tensor,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<Quantity<Energy>, Self::Error>;
}
