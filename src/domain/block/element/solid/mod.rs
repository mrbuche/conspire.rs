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
