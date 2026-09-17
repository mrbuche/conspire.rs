use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    domain::block::element::solid::elastic::ElasticElement, math::Quantity, units::Energy,
};

pub trait HyperelasticElement<C>
where
    C: Hyperelastic,
    Self: ElasticElement<C>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
    ) -> Result<Quantity<Energy>, Self::Error>;
}
