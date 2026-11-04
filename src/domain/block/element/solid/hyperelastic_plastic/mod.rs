use crate::{
    constitutive::solid::hyperelastic_plastic::HyperelasticPlastic,
    domain::block::element::solid::{
        elastic_plastic::ElasticPlasticElement, plastic::PlasticStateVariables,
    },
    math::Quantity,
    units::Energy,
};

pub trait HyperelasticPlasticElement<C, const G: usize>
where
    C: HyperelasticPlastic,
    Self: ElasticPlasticElement<C, G>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        state_variables: &PlasticStateVariables<G>,
    ) -> Result<Quantity<Energy>, Self::Error>;
}
