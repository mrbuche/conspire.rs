use crate::{
    constitutive::solid::hyperviscoelastic::Hyperviscoelastic,
    domain::block::element::solid::elastic_hyperviscous::ElasticHyperviscousElement,
    math::Quantity, units::Energy,
};

pub trait HyperviscoelasticElement<C, const P: usize>
where
    C: Hyperviscoelastic,
    Self: ElasticHyperviscousElement<C, P>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
    ) -> Result<Quantity<Energy>, Self::Error>;
}
