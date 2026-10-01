pub(crate) mod elastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod elastic_hyperviscous;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod elastic_plastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod elastic_viscoplastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod hyperelastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod hyperelastic_viscoplastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod hyperviscoelastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod plastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod viscoelastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod viscoplastic;

pub trait SolidElement {
    type Coordinates;
    type Velocities;
    type DeformationGradients;
    type DeformationGradientRates;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &Self::Coordinates,
    ) -> Self::DeformationGradients;
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &Self::Coordinates,
        nodal_velocities: &Self::Velocities,
    ) -> Self::DeformationGradientRates;
}
