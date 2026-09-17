use crate::{
    constitutive::solid::viscoelastic::Viscoelastic, domain::block::element::solid::SolidElement,
};

pub trait ViscoelasticElement<C, const P: usize>
where
    C: Viscoelastic,
    Self: SolidElement,
{
    type Forces;
    type Dampings;
    type Error;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        nodal_velocities: &Self::Velocities,
    ) -> Result<Self::Forces, Self::Error>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
        nodal_velocities: &Self::Velocities,
    ) -> Result<Self::Dampings, Self::Error>;
}
