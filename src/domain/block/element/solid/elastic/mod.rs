use crate::{constitutive::solid::elastic::Elastic, domain::block::element::solid::SolidElement};

pub trait ElasticElement<C>
where
    C: Elastic,
    Self: SolidElement,
{
    type Forces;
    type Stiffnesses;
    type Error;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
    ) -> Result<Self::Forces, Self::Error>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &Self::Coordinates,
    ) -> Result<Self::Stiffnesses, Self::Error>;
}
