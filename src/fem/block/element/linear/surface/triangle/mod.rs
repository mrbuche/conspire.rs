#[cfg(test)]
mod test;

use super::*;

const G: usize = 1;
const M: usize = 2;
const N: usize = 3;
const O: usize = 3;

const INTEGRATION_WEIGHT: Scalar = 0.5;

pub const STANDARD_GRADIENT_OPERATOR: StandardGradientOperator<M, O> = TensorRank1List([
    TensorRank1([-1.0, -1.0]),
    TensorRank1([1.0, 0.0]),
    TensorRank1([0.0, 1.0]),
]);

pub struct Triangle<C> {
    constitutive_model: C,
    gradient_vectors: GradientVectors<N>,
    integration_weight: Scalar,
    reference_normal: ReferenceNormal,
}

impl<'a, C> SurfaceElement<'a, C, G, N> for Triangle<C>
where
    C: Constitutive<'a>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
        thickness: &Scalar,
    ) -> Self {
        Self {
            constitutive_model: <C>::new(constitutive_model_parameters),
            gradient_vectors: Self::calculate_gradient_vectors(&reference_nodal_coordinates),
            integration_weight: Self::calculate_reference_jacobian(&reference_nodal_coordinates)
                * INTEGRATION_WEIGHT
                * thickness,
            reference_normal: Self::calculate_reference_normal(&reference_nodal_coordinates),
        }
    }
}

impl<'a, C> LinearElement<'a, C, G, M, N, O> for Triangle<C>
where
    C: Constitutive<'a>,
{
    fn calculate_deformation_gradient(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradient {
        self.calculate_deformation_gradient_linear_surface_element(nodal_coordinates)
    }
    fn calculate_deformation_gradient_rate(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRate {
        self.calculate_deformation_gradient_rate_linear_surface_element(
            nodal_coordinates,
            nodal_velocities,
        )
    }
    fn calculate_gradient_vectors(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> GradientVectors<N> {
        Self::calculate_gradient_vectors_linear_surface_element(reference_nodal_coordinates)
    }
    fn calculate_reference_jacobian(
        reference_nodal_coordinates: &ReferenceNodalCoordinates<O>,
    ) -> Scalar {
        Self::calculate_reference_jacobian_linear_surface_element(reference_nodal_coordinates)
    }
    fn calculate_standard_gradient_operator() -> StandardGradientOperator<M, O> {
        STANDARD_GRADIENT_OPERATOR
    }
    fn get_constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn get_gradient_vectors(&self) -> &GradientVectors<N> {
        &self.gradient_vectors
    }
    fn get_integration_weight(&self) -> &Scalar {
        &self.integration_weight
    }
}

impl<'a, C> ElasticFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: Elastic<'a>,
{
    fn calculate_deformations(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradients<G> {
        TensorRank2List([self.calculate_deformation_gradient(nodal_coordinates)])
    }
    fn calculate_nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        self.calculate_nodal_forces_linear_element(nodal_coordinates)
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        let first_piola_kirchoff_tangent_stiffness = self
            .get_constitutive_model()
            .calculate_first_piola_kirchoff_tangent_stiffness(
                &self.calculate_deformation_gradient(nodal_coordinates),
            )?;
        let gradient_vectors = self.get_gradient_vectors();
        let normal_gradients = Self::calculate_normal_gradients(nodal_coordinates);
        let reference_normal = self.get_reference_normal();
        Ok(gradient_vectors.iter()
        .map(|gradient_vector_a|
            gradient_vectors.iter()
            .zip(normal_gradients.iter())
            .map(|(gradient_vector_b, normal_gradient_b)|
                first_piola_kirchoff_tangent_stiffness.iter()
                .map(|first_piola_kirchoff_tangent_stiffness_m|
                    IDENTITY.iter()
                    .zip(normal_gradient_b.iter())
                    .map(|(identity_n, normal_gradient_b_n)|
                        first_piola_kirchoff_tangent_stiffness_m.iter()
                        .zip(gradient_vector_a.iter())
                        .map(|(first_piola_kirchoff_tangent_stiffness_mj, gradient_vector_a_j)|
                            first_piola_kirchoff_tangent_stiffness_mj.iter()
                            .zip(identity_n.iter()
                            .zip(normal_gradient_b_n.iter()))
                            .map(|(first_piola_kirchoff_tangent_stiffness_mjk, (identity_nk, normal_gradient_b_n_k))|
                                first_piola_kirchoff_tangent_stiffness_mjk.iter()
                                .zip(gradient_vector_b.iter()
                                .zip(reference_normal.iter()))
                                .map(|(first_piola_kirchoff_tangent_stiffness_mjkl, (gradient_vector_b_l, reference_normal_l))|
                                    first_piola_kirchoff_tangent_stiffness_mjkl * gradient_vector_a_j * (
                                        identity_nk * gradient_vector_b_l + normal_gradient_b_n_k * reference_normal_l
                                    ) * self.get_integration_weight()
                                ).sum::<Scalar>()
                            ).sum::<Scalar>()
                        ).sum::<Scalar>()
                    ).collect()
                ).collect()
            ).collect()
        ).collect())
    }
}

impl<'a, C> ViscoelasticFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: Viscoelastic<'a>,
{
    fn calculate_nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        self.calculate_nodal_forces_linear_element(nodal_coordinates, nodal_velocities)
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        let first_piola_kirchoff_rate_tangent_stiffness = self
            .get_constitutive_model()
            .calculate_first_piola_kirchoff_rate_tangent_stiffness(
                &self.calculate_deformation_gradient(nodal_coordinates),
                &self.calculate_deformation_gradient_rate(nodal_coordinates, nodal_velocities),
            )?;
        let gradient_vectors = self.get_gradient_vectors();
        let normal_gradients = Self::calculate_normal_gradients(nodal_coordinates);
        let reference_normal = self.get_reference_normal();
        Ok(gradient_vectors.iter()
        .map(|gradient_vector_a|
            gradient_vectors.iter()
            .zip(normal_gradients.iter())
            .map(|(gradient_vector_b, normal_gradient_b)|
                first_piola_kirchoff_rate_tangent_stiffness.iter()
                .map(|first_piola_kirchoff_rate_tangent_stiffness_m|
                    IDENTITY.iter()
                    .zip(normal_gradient_b.iter())
                    .map(|(identity_n, normal_gradient_b_n)|
                        first_piola_kirchoff_rate_tangent_stiffness_m.iter()
                        .zip(gradient_vector_a.iter())
                        .map(|(first_piola_kirchoff_rate_tangent_stiffness_mj, gradient_vector_a_j)|
                            first_piola_kirchoff_rate_tangent_stiffness_mj.iter()
                            .zip(identity_n.iter()
                            .zip(normal_gradient_b_n.iter()))
                            .map(|(first_piola_kirchoff_rate_tangent_stiffness_mjk, (identity_nk, normal_gradient_b_n_k))|
                                first_piola_kirchoff_rate_tangent_stiffness_mjk.iter()
                                .zip(gradient_vector_b.iter()
                                .zip(reference_normal.iter()))
                                .map(|(first_piola_kirchoff_rate_tangent_stiffness_mjkl, (gradient_vector_b_l, reference_normal_l))|
                                    first_piola_kirchoff_rate_tangent_stiffness_mjkl * gradient_vector_a_j * (
                                        identity_nk * gradient_vector_b_l + normal_gradient_b_n_k * reference_normal_l
                                    ) * self.get_integration_weight()
                                ).sum::<Scalar>()
                            ).sum::<Scalar>()
                        ).sum::<Scalar>()
                    ).collect()
                ).collect()
            ).collect()
        ).collect())
    }
}

impl<'a, C> LinearSurfaceElement<'a, C, G, M, N, O> for Triangle<C>
where
    C: Constitutive<'a>,
{
    fn get_reference_normal(&self) -> &ReferenceNormal {
        &self.reference_normal
    }
}

impl<'a, C> ElasticLinearElement<'a, C, G, M, N, O> for Triangle<C> where C: Elastic<'a> {}

impl<'a, C> HyperelasticFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: Hyperelastic<'a>,
{
    fn calculate_helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_helmholtz_free_energy_linear_element(nodal_coordinates)
    }
}

impl<'a, C> HyperelasticLinearElement<'a, C, G, M, N, O> for Triangle<C> where C: Hyperelastic<'a> {}

impl<'a, C> ViscoelasticLinearElement<'a, C, G, M, N, O> for Triangle<C> where C: Viscoelastic<'a> {}

impl<'a, C> ElasticHyperviscousFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: ElasticHyperviscous<'a>,
{
    fn calculate_viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_viscous_dissipation_linear_element(nodal_coordinates, nodal_velocities)
    }
    fn calculate_dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_dissipation_potential_linear_element(nodal_coordinates, nodal_velocities)
    }
}

impl<'a, C> ElasticHyperviscousLinearElement<'a, C, G, M, N, O> for Triangle<C> where
    C: ElasticHyperviscous<'a>
{
}

impl<'a, C> HyperviscoelasticFiniteElement<'a, C, G, N> for Triangle<C>
where
    C: Hyperviscoelastic<'a>,
{
    fn calculate_helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.calculate_helmholtz_free_energy_linear_element(nodal_coordinates)
    }
}

impl<'a, C> HyperviscoelasticLinearElement<'a, C, G, M, N, O> for Triangle<C> where
    C: Hyperviscoelastic<'a>
{
}
