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
    /// The nodal forces and the nodal stiffnesses, with the plastic step of every
    /// integration point solved once for both.
    fn nodal_forces_and_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<(Self::Forces, Self::Stiffnesses), Self::Error>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<Self::Forces, Self::Error> {
        Ok(self
            .nodal_forces_and_stiffnesses(constitutive_model, nodal_coordinates, state_variables)?
            .0)
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<Self::Stiffnesses, Self::Error> {
        Ok(self
            .nodal_forces_and_stiffnesses(constitutive_model, nodal_coordinates, state_variables)?
            .1)
    }
    fn updated_state(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<PlasticStateVariables<G>, Self::Error>;
}
