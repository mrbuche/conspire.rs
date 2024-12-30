#[cfg(test)]
mod test;

pub mod wedge;

use super::{surface::*, *};

pub trait LinearLocalizationElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
> where
    C: Constitutive<'a>,
    Self: LinearSurfaceElement<'a, C, G, M, N, O>,
{
    fn calculate_deformation_gradient_linear_localization_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradient {
        nodal_coordinates
            .iter()
            .zip(self.get_gradient_vectors().iter())
            .map(|(nodal_coordinate, gradient_vector)| {
                DeformationGradient::dyad(nodal_coordinate, gradient_vector)
            })
            .sum::<DeformationGradient>()
            + DeformationGradient::dyad(
                &Self::calculate_normal(&Self::calculate_midplane(nodal_coordinates)),
                self.get_reference_normal(),
            )
    }
    fn calculate_deformation_gradient_rate_linear_localization_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRate {
        nodal_velocities
            .iter()
            .zip(self.get_gradient_vectors().iter())
            .map(|(nodal_velocity, gradient_vector)| {
                DeformationGradientRate::dyad(nodal_velocity, gradient_vector)
            })
            .sum::<DeformationGradientRate>()
            + DeformationGradientRate::dyad(
                &Self::calculate_normal_rate(
                    &Self::calculate_midplane(nodal_coordinates),
                    &Self::calculate_midplane(nodal_velocities),
                ),
                self.get_reference_normal(),
            )
    }
    fn calculate_gradient_vectors_linear_localization_element(
        reference_nodal_coordinates_midplane: &ReferenceNodalCoordinates<O>,
        thickness: &Scalar,
    ) -> GradientVectors<N>;
    fn calculate_midplane<const I: usize>(
        nodal_coordinates: &Coordinates<I, N>,
    ) -> Coordinates<I, O>;
}
