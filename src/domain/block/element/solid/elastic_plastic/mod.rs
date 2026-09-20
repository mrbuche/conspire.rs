use crate::{
    constitutive::solid::elastic_plastic::ElasticPlastic,
    domain::block::element::solid::{SolidElement, plastic::PlasticStateVariables},
};

pub trait ElasticPlasticElement<C, const G: usize>
where
    C: ElasticPlastic,
    Self: SolidElement,
{
    type Forces;
    type Stiffnesses;
    type Error;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<Self::Forces, Self::Error>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<Self::Stiffnesses, Self::Error>;
    fn updated_state(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<PlasticStateVariables<G>, Self::Error>;
}
