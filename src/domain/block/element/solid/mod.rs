pub(crate) mod elastic;
pub(crate) mod elastic_hyperviscous;
pub(crate) mod elastic_viscoplastic;
pub(crate) mod hyperelastic;
pub(crate) mod hyperelastic_viscoplastic;
pub(crate) mod hyperviscoelastic;
pub(crate) mod viscoelastic;
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
