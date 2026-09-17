use crate::{
    constitutive::solid::elastic_viscoplastic::ElasticViscoplastic,
    domain::block::element::solid::{
        SolidElement,
        viscoplastic::{ViscoplasticEvolution, ViscoplasticStateVariables},
    },
    math::{Differentiable, Tensor},
};

pub trait ElasticViscoplasticElement<C, const G: usize, Y>
where
    C: ElasticViscoplastic<Y>,
    Self: SolidElement,
    Y: Differentiable + Tensor,
{
    type Forces;
    type Stiffnesses;
    type Error;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<Self::Forces, Self::Error>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<Self::Stiffnesses, Self::Error>;
    fn state_variables_evolution(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<ViscoplasticEvolution<G, Y>, Self::Error>;
}
