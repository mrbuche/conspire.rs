use crate::{
    constitutive::solid::elastic_hyperviscous::ElasticHyperviscous,
    domain::block::element::solid::viscoelastic::ViscoelasticElement, math::Quantity, units::Power,
};

pub trait ElasticHyperviscousElement<C, const P: usize>
where
    C: ElasticHyperviscous,
    Self: ViscoelasticElement<C, P>,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        nodal_velocities: &Self::Velocities,
    ) -> Result<Quantity<Power>, Self::Error>;
    fn dissipation_potential(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        nodal_velocities: &Self::Velocities,
    ) -> Result<Quantity<Power>, Self::Error>;
}
