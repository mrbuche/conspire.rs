#[cfg(test)]
pub mod test;

pub mod cohesive;
pub mod localization;
pub mod surface;
pub mod tetrahedron;

use super::*;

pub trait LinearElement<'a, C, const G: usize, const M: usize, const N: usize, const O: usize>
where
    C: Constitutive<'a>,
{
    fn calculate_deformation_gradient(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradient {
        nodal_coordinates
            .iter()
            .zip(self.get_gradient_vectors().iter())
            .map(|(nodal_coordinate, gradient_vector)| {
                DeformationGradient::dyad(nodal_coordinate, gradient_vector)
            })
            .sum()
    }
    fn calculate_deformation_gradient_rate(
        &self,
        _: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRate {
        nodal_velocities
            .iter()
            .zip(self.get_gradient_vectors().iter())
            .map(|(nodal_velocity, gradient_vector)| {
                DeformationGradientRate::dyad(nodal_velocity, gradient_vector)
            })
            .sum()
    }
    fn calculate_gradient_vectors(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> GradientVectors<N>;
    fn calculate_reference_jacobian(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> Scalar;
    fn calculate_standard_gradient_operator() -> StandardGradientOperator<M, O>;
    fn get_constitutive_model(&self) -> &C;
    fn get_gradient_vectors(&self) -> &GradientVectors<N>;
    fn get_integration_weight(&self) -> &Scalar;
}

pub trait ElasticLinearElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
> where
    C: Elastic<'a>,
    Self: LinearElement<'a, C, G, M, N, O>,
{
    fn calculate_nodal_forces_linear_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        let first_piola_kirchoff_stress = self
            .get_constitutive_model()
            .calculate_first_piola_kirchoff_stress(
                &self.calculate_deformation_gradient(nodal_coordinates),
            )?;
        Ok(self
            .get_gradient_vectors()
            .iter()
            .map(|gradient_vector| {
                (&first_piola_kirchoff_stress * gradient_vector) * self.get_integration_weight()
            })
            .collect())
    }
    fn calculate_nodal_stiffnesses_linear_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        let first_piola_kirchoff_tangent_stiffness = self
            .get_constitutive_model()
            .calculate_first_piola_kirchoff_tangent_stiffness(
                &self.calculate_deformation_gradient(nodal_coordinates),
            )?;
        let gradient_vectors = self.get_gradient_vectors();
        Ok(gradient_vectors
            .iter()
            .map(|gradient_vector_a| {
                gradient_vectors
                    .iter()
                    .map(|gradient_vector_b| {
                        first_piola_kirchoff_tangent_stiffness
                            .contract_second_fourth_indices_with_first_indices_of(
                                gradient_vector_a,
                                gradient_vector_b,
                            )
                            * self.get_integration_weight()
                    })
                    .collect()
            })
            .collect())
    }
}

pub trait HyperelasticLinearElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
> where
    C: Hyperelastic<'a>,
    Self: ElasticLinearElement<'a, C, G, M, N, O>,
{
    fn calculate_helmholtz_free_energy_linear_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self
            .get_constitutive_model()
            .calculate_helmholtz_free_energy_density(
                &self.calculate_deformation_gradient(nodal_coordinates),
            )?
            * self.get_integration_weight())
    }
}

pub trait ViscoelasticLinearElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
> where
    C: Viscoelastic<'a>,
    Self: LinearElement<'a, C, G, M, N, O>,
{
    fn calculate_nodal_forces_linear_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        let first_piola_kirchoff_stress = self
            .get_constitutive_model()
            .calculate_first_piola_kirchoff_stress(
                &self.calculate_deformation_gradient(nodal_coordinates),
                &self.calculate_deformation_gradient_rate(nodal_coordinates, nodal_velocities),
            )?;
        Ok(self
            .get_gradient_vectors()
            .iter()
            .map(|gradient_vector| {
                (&first_piola_kirchoff_stress * gradient_vector) * self.get_integration_weight()
            })
            .collect())
    }
    fn calculate_nodal_stiffnesses_linear_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        let first_piola_kirchoff_tangent_stiffness = self
            .get_constitutive_model()
            .calculate_first_piola_kirchoff_rate_tangent_stiffness(
                &self.calculate_deformation_gradient(nodal_coordinates),
                &self.calculate_deformation_gradient_rate(nodal_coordinates, nodal_velocities),
            )?;
        let gradient_vectors = self.get_gradient_vectors();
        Ok(gradient_vectors
            .iter()
            .map(|gradient_vector_a| {
                gradient_vectors
                    .iter()
                    .map(|gradient_vector_b| {
                        first_piola_kirchoff_tangent_stiffness
                            .contract_second_fourth_indices_with_first_indices_of(
                                gradient_vector_a,
                                gradient_vector_b,
                            )
                            * self.get_integration_weight()
                    })
                    .collect()
            })
            .collect())
    }
}

pub trait ElasticHyperviscousLinearElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
> where
    C: ElasticHyperviscous<'a>,
    Self: ViscoelasticLinearElement<'a, C, G, M, N, O>,
{
    fn calculate_viscous_dissipation_linear_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self
            .get_constitutive_model()
            .calculate_viscous_dissipation(
                &self.calculate_deformation_gradient(nodal_coordinates),
                &self.calculate_deformation_gradient_rate(nodal_coordinates, nodal_velocities),
            )?
            * self.get_integration_weight())
    }
    fn calculate_dissipation_potential_linear_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self
            .get_constitutive_model()
            .calculate_dissipation_potential(
                &self.calculate_deformation_gradient(nodal_coordinates),
                &self.calculate_deformation_gradient_rate(nodal_coordinates, nodal_velocities),
            )?
            * self.get_integration_weight())
    }
}

pub trait HyperviscoelasticLinearElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
> where
    C: Hyperviscoelastic<'a>,
    Self: ElasticHyperviscousLinearElement<'a, C, G, M, N, O>,
{
    fn calculate_helmholtz_free_energy_linear_element(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self
            .get_constitutive_model()
            .calculate_helmholtz_free_energy_density(
                &self.calculate_deformation_gradient(nodal_coordinates),
            )?
            * self.get_integration_weight())
    }
}
