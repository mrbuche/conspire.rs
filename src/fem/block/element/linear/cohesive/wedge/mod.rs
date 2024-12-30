#[cfg(test)]
pub mod test;

use super::*;

const G: usize = 1;
const M: usize = 2;
const N: usize = 6;
const O: usize = 3;

const INTEGRATION_WEIGHT: Scalar = 1.0 / 2.0;

pub struct Wedge<C> {
    constitutive_model: C,
    integration_weight: Scalar,
}

impl<'a, C> FiniteElement<'a, C, G, N> for Wedge<C>
where
    C: Cohesive<'a>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self {
        let reference_nodal_coordinates_midplane =
            Self::calculate_midplane(&reference_nodal_coordinates);
        Self {
            constitutive_model: <C>::new(constitutive_model_parameters),
            integration_weight: Self::calculate_reference_jacobian(
                &reference_nodal_coordinates_midplane,
            ) * INTEGRATION_WEIGHT,
        }
    }
}

impl<'a, C> LinearElement<'a, C, G, M, N, O> for Wedge<C>
where
    C: Cohesive<'a>,
{
    fn calculate_deformation_gradient(&self, _: &NodalCoordinates<N>) -> DeformationGradient {
        panic!()
    }
    fn calculate_deformation_gradient_rate(
        &self,
        _: &NodalCoordinates<N>,
        _: &NodalVelocities<N>,
    ) -> DeformationGradientRate {
        panic!()
    }
    fn calculate_gradient_vectors(_: &ReferenceNodalCoordinates<O>) -> GradientVectors<N> {
        panic!()
    }
    fn calculate_reference_jacobian(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> Scalar {
        Self::calculate_reference_jacobian_linear_surface_element(reference_nodal_coordinates)
    }
    fn calculate_standard_gradient_operator() -> StandardGradientOperator<M, O> {
        StandardGradientOperator::new([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    }
    fn get_constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn get_gradient_vectors(&self) -> &GradientVectors<N> {
        panic!()
    }
    fn get_integration_weight(&self) -> &Scalar {
        &self.integration_weight
    }
}

impl<'a, C> LinearSurfaceElement<'a, C, G, M, N, O> for Wedge<C>
where
    C: Cohesive<'a>,
{
    fn calculate_deformation_gradient_linear_surface_element(
        &self,
        _: &NodalCoordinates<O>,
    ) -> DeformationGradient {
        panic!()
    }
    fn calculate_deformation_gradient_rate_linear_surface_element(
        &self,
        _: &NodalCoordinates<O>,
        _: &NodalVelocities<O>,
    ) -> DeformationGradientRate {
        panic!()
    }
    fn calculate_gradient_vectors_linear_surface_element(
        _: &ReferenceNodalCoordinates<O>,
    ) -> GradientVectors<N> {
        panic!()
    }
    fn get_reference_normal(&self) -> &ReferenceNormal {
        panic!()
    }
}

impl<'a, C> CohesiveElement<'a, C, G, N> for Wedge<C>
where
    C: Cohesive<'a>,
{
    fn calculate_nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        let scaled_traction = self.get_constitutive_model().calculate_traction(
            &Self::calculate_displacement(nodal_coordinates),
            &Self::calculate_normal(&Self::calculate_midplane(nodal_coordinates)),
        ) * (self.get_integration_weight() / 3.0);
        Ok((0..N)
            .map(|node| &scaled_traction * (1.0 - 2.0 * (((node >= O) as u8) as Scalar)))
            .collect())
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        let midplane = Self::calculate_midplane(nodal_coordinates);
        let (stiffness_opening, stiffness_normal) =
            self.get_constitutive_model().calculate_stiffnesses(
                &Self::calculate_displacement(nodal_coordinates),
                &Self::calculate_normal(&midplane),
            );
        let objects = Self::calculate_normal_gradients(&midplane)
            .iter()
            .map(|normal_gradient_b| {
                stiffness_normal
                    .iter()
                    .map(|stiffness_normal_i| {
                        normal_gradient_b
                            .iter()
                            .map(|normal_gradient_b_k| stiffness_normal_i * normal_gradient_b_k)
                            .collect()
                    })
                    .collect()
            })
            .collect::<NormalGradients<O>>();
        Ok((0..N)
            .map(|node_a| {
                (0..N)
                    .zip(objects.iter().chain(objects.iter()))
                    .map(|(node_b, object_b)| {
                        &stiffness_opening
                            * (self.get_integration_weight() / -9.0
                                * (1.0 - 2.0 * (((node_a >= O) as u8) as Scalar))
                                * (1.0 - 2.0 * (((node_b >= O) as u8) as Scalar)))
                            + object_b
                                * (self.get_integration_weight() / 6.0
                                    * (1.0 - 2.0 * (((node_a >= O) as u8) as Scalar)))
                    })
                    .collect()
            })
            .collect())
    }
}

impl<'a, C> LinearCohesiveElement<'a, C, G, M, N, O> for Wedge<C>
where
    C: Cohesive<'a>,
{
    fn calculate_displacement(nodal_coordinates: &NodalCoordinates<N>) -> Displacement {
        nodal_coordinates
            .iter()
            .skip(O)
            .zip(nodal_coordinates.iter().take(O))
            .map(|(nodal_coordinates_top, nodal_coordinates_bottom)| {
                nodal_coordinates_top
                    .iter()
                    .zip(nodal_coordinates_bottom.iter())
                    .map(|(nodal_coordinates_top_i, nodal_coordinates_bottom_i)| {
                        (nodal_coordinates_top_i - nodal_coordinates_bottom_i) / 3.0
                    })
                    .collect()
            })
            .sum()
    }
    fn calculate_midplane<const I: usize>(
        nodal_coordinates: &Coordinates<I, N>,
    ) -> Coordinates<I, O> {
        nodal_coordinates
            .iter()
            .skip(O)
            .zip(nodal_coordinates.iter().take(O))
            .map(|(nodal_coordinates_top, nodal_coordinates_bottom)| {
                nodal_coordinates_top
                    .iter()
                    .zip(nodal_coordinates_bottom.iter())
                    .map(|(nodal_coordinates_top_i, nodal_coordinates_bottom_i)| {
                        (nodal_coordinates_top_i + nodal_coordinates_bottom_i) * 0.5
                    })
                    .collect()
            })
            .collect()
    }
}
