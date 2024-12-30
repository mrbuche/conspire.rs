#[cfg(test)]
mod test;

pub mod wedge;

use super::{surface::*, *};

pub trait CompositeLocalizationElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
> where
    C: Constitutive<'a>,
    Self: CompositeSurfaceElement<'a, C, G, M, N, O, P, Q>,
{
    fn calculate_deformation_gradients_composite_localization_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradients<G> {
        let normals = Self::calculate_normals(&Self::calculate_midplane(nodal_coordinates));
        self.get_projected_gradient_vectors()
            .iter()
            .zip(self.get_scaled_reference_normals().iter())
            .map(|(projected_gradient_vectors, scaled_reference_normals)| {
                nodal_coordinates
                    .iter()
                    .zip(projected_gradient_vectors.iter())
                    .map(|(nodal_coordinate, projected_gradient_vector)| {
                        DeformationGradient::dyad(nodal_coordinate, projected_gradient_vector)
                    })
                    .sum::<DeformationGradient>()
                    + scaled_reference_normals
                        .iter()
                        .zip(normals.iter())
                        .map(|(scaled_reference_normal, normal)| {
                            DeformationGradient::dyad(normal, scaled_reference_normal)
                        })
                        .sum::<DeformationGradient>()
            })
            .collect()
    }
    fn calculate_deformation_gradient_rates_composite_localization_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRates<G> {
        let normal_rates = Self::calculate_normal_rates(
            &Self::calculate_midplane(nodal_coordinates),
            &Self::calculate_midplane(nodal_velocities),
        );
        self.get_projected_gradient_vectors()
            .iter()
            .zip(self.get_scaled_reference_normals().iter())
            .map(|(projected_gradient_vectors, scaled_reference_normals)| {
                nodal_velocities
                    .iter()
                    .zip(projected_gradient_vectors.iter())
                    .map(|(nodal_velocity, projected_gradient_vector)| {
                        DeformationGradientRate::dyad(nodal_velocity, projected_gradient_vector)
                    })
                    .sum::<DeformationGradientRate>()
                    + scaled_reference_normals
                        .iter()
                        .zip(normal_rates.iter())
                        .map(|(scaled_reference_normal, normal_rate)| {
                            DeformationGradientRate::dyad(normal_rate, scaled_reference_normal)
                        })
                        .sum::<DeformationGradientRate>()
            })
            .collect()
    }
    fn calculate_midplane<const I: usize>(
        nodal_coordinates: &Coordinates<I, N>,
    ) -> Coordinates<I, O>;
    fn calculate_mixed_shape_function_integrals_products(
    ) -> ShapeFunctionIntegralsProductsMixed<O, P>;
    fn calculate_projected_gradient_vectors_composite_localization_element(
        reference_nodal_coordinates_midplane: &ReferenceNodalCoordinates<O>,
        thickness: &Scalar,
    ) -> ProjectedGradientVectors<G, N>;
}
