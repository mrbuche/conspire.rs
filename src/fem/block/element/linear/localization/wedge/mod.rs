#[cfg(test)]
mod test;

use super::{super::surface::triangle::STANDARD_GRADIENT_OPERATOR, *};

const G: usize = 1;
const M: usize = 2;
const N: usize = 6;
const O: usize = 3;

const INTEGRATION_WEIGHT: Scalar = 0.5;

pub struct Wedge<C> {
    constitutive_model: C,
    gradient_vectors: GradientVectors<N>,
    integration_weight: Scalar,
    reference_normal: ReferenceNormal,
}

impl<'a, C> SurfaceElement<'a, C, G, N> for Wedge<C>
where
    C: Constitutive<'a>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
        thickness: &Scalar,
    ) -> Self {
        let reference_nodal_coordinates_midplane =
            Self::calculate_midplane(&reference_nodal_coordinates);
        Self {
            constitutive_model: <C>::new(constitutive_model_parameters),
            gradient_vectors: Self::calculate_gradient_vectors_linear_localization_element(
                &reference_nodal_coordinates_midplane,
                thickness,
            ),
            integration_weight: Self::calculate_reference_jacobian(
                &reference_nodal_coordinates_midplane,
            ) * INTEGRATION_WEIGHT
                * thickness,
            reference_normal: Self::calculate_reference_normal(
                &reference_nodal_coordinates_midplane,
            ),
        }
    }
}

impl<'a, C> LinearElement<'a, C, G, M, N, O> for Wedge<C>
where
    C: Constitutive<'a>,
{
    fn calculate_deformation_gradient(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradient {
        self.calculate_deformation_gradient_linear_localization_element(nodal_coordinates)
    }
    fn calculate_deformation_gradient_rate(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRate {
        self.calculate_deformation_gradient_rate_linear_localization_element(
            nodal_coordinates,
            nodal_velocities,
        )
    }
    fn calculate_gradient_vectors(_: &ReferenceNodalCoordinates<O>) -> GradientVectors<N> {
        panic!()
    }
    fn calculate_reference_jacobian(
        reference_nodal_coordinates_midplane: &ReferenceNodalCoordinates<O>,
    ) -> Scalar {
        Self::calculate_reference_jacobian_linear_surface_element(
            reference_nodal_coordinates_midplane,
        )
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

impl<'a, C> LinearLocalizationElement<'a, C, G, M, N, O> for Wedge<C>
where
    C: Constitutive<'a>,
{
    fn calculate_gradient_vectors_linear_localization_element(
        reference_nodal_coordinates_midplane: &ReferenceNodalCoordinates<O>,
        thickness: &Scalar,
    ) -> GradientVectors<N> {
        let reference_dual_basis_vectors =
            Self::calculate_dual_basis(reference_nodal_coordinates_midplane);
        let scaled_reference_normal =
            Self::calculate_reference_normal(reference_nodal_coordinates_midplane)
                / (3.0 * thickness);
        let gradient_vectors_midplane = STANDARD_GRADIENT_OPERATOR
            .iter()
            .map(|standard_gradient_operator_a| {
                standard_gradient_operator_a
                    .iter()
                    .zip(reference_dual_basis_vectors.iter())
                    .map(
                        |(standard_gradient_operator_a_m, dual_reference_basis_vector_m)| {
                            dual_reference_basis_vector_m * standard_gradient_operator_a_m
                        },
                    )
                    .sum()
            })
            .collect::<GradientVectors<O>>();
        let mut gradient_vectors = GradientVectors::zero();
        gradient_vectors
            .iter_mut()
            .skip(O)
            .zip(gradient_vectors_midplane.iter())
            .for_each(|(gradient_vector_a, gradient_vector_midplane_a)| {
                *gradient_vector_a = gradient_vector_midplane_a * 0.5 + &scaled_reference_normal
            });
        gradient_vectors
            .iter_mut()
            .take(O)
            .zip(gradient_vectors_midplane.iter())
            .for_each(|(gradient_vector_a, gradient_vector_midplane_a)| {
                *gradient_vector_a = gradient_vector_midplane_a * 0.5 - &scaled_reference_normal
            });
        gradient_vectors
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

impl<'a, C> ElasticFiniteElement<'a, C, G, N> for Wedge<C>
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
        let first_piola_kirchoff_stress = self
            .get_constitutive_model()
            .calculate_first_piola_kirchoff_stress(
                &self.calculate_deformation_gradient(nodal_coordinates),
            )?;
        let normal_gradients =
            Self::calculate_normal_gradients(&Self::calculate_midplane(nodal_coordinates));
        let traction = (&first_piola_kirchoff_stress * self.get_reference_normal()) * 0.5;
        Ok(self
            .get_gradient_vectors()
            .iter()
            .zip(normal_gradients.iter().chain(normal_gradients.iter()))
            .map(|(gradient_vector_a, normal_gradient_a)| {
                (&first_piola_kirchoff_stress * gradient_vector_a + normal_gradient_a * &traction)
                    * self.get_integration_weight()
            })
            .collect())
    }
    fn calculate_nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        let deformation_gradient = self.calculate_deformation_gradient(nodal_coordinates);
        let first_piola_kirchoff_stress = self
            .get_constitutive_model()
            .calculate_first_piola_kirchoff_stress(&deformation_gradient)?;
        let first_piola_kirchoff_tangent_stiffness = self
            .get_constitutive_model()
            .calculate_first_piola_kirchoff_tangent_stiffness(&deformation_gradient)?;
        let gradient_vectors = self.get_gradient_vectors();
        let midplane = Self::calculate_midplane(nodal_coordinates);
        let normal_gradients = Self::calculate_normal_gradients(&midplane);
        let normal_tangents = Self::calculate_normal_tangents(&midplane);
        let reference_normal = self.get_reference_normal() * 0.5;
        let traction = (first_piola_kirchoff_stress * &reference_normal) * 0.5;
        Ok(gradient_vectors.iter()
        .zip(normal_gradients.iter()
        .chain(normal_gradients.iter()))
        .map(|(gradient_vector_a, normal_gradient_a)|
            gradient_vectors.iter()
            .zip(normal_gradients.iter()
            .chain(normal_gradients.iter()))
            .map(|(gradient_vector_b, normal_gradient_b)|
                IDENTITY.iter()
                .zip(normal_gradient_a.iter())
                .map(|(identity_m, normal_gradient_a_m)|
                    IDENTITY.iter()
                    .zip(normal_gradient_b.iter())
                    .map(|(identity_n, normal_gradient_b_n)|
                        first_piola_kirchoff_tangent_stiffness.iter()
                        .zip(identity_m.iter()
                        .zip(normal_gradient_a_m.iter()))
                        .map(|(first_piola_kirchoff_tangent_stiffness_i, (identity_mi, normal_gradient_a_m_i))|
                            first_piola_kirchoff_tangent_stiffness_i.iter()
                            .zip(gradient_vector_a.iter()
                            .zip(reference_normal.iter()))
                            .map(|(first_piola_kirchoff_tangent_stiffness_ij, (gradient_vector_a_j, reference_normal_j))|
                                first_piola_kirchoff_tangent_stiffness_ij.iter()
                                .zip(identity_n.iter()
                                .zip(normal_gradient_b_n.iter()))
                                .map(|(first_piola_kirchoff_tangent_stiffness_ijk, (identity_nk, normal_gradient_b_n_k))|
                                    first_piola_kirchoff_tangent_stiffness_ijk.iter()
                                    .zip(gradient_vector_b.iter()
                                    .zip(reference_normal.iter()))
                                    .map(|(first_piola_kirchoff_tangent_stiffness_ijkl, (gradient_vector_b_l, reference_normal_l))|
                                        first_piola_kirchoff_tangent_stiffness_ijkl * (
                                            identity_mi * gradient_vector_a_j + normal_gradient_a_m_i * reference_normal_j
                                        ) * (
                                            identity_nk * gradient_vector_b_l + normal_gradient_b_n_k * reference_normal_l
                                        ) * self.get_integration_weight()
                                    ).sum::<Scalar>()
                                ).sum::<Scalar>()
                            ).sum::<Scalar>()
                        ).sum::<Scalar>()
                    ).collect()
                ).collect()
            ).collect()
        ).collect::<NodalStiffnesses<N>>() +
        normal_tangents.iter()
        .chain(normal_tangents.iter())
        .map(|normal_tangent_a|
            normal_tangent_a.iter()
            .chain(normal_tangent_a.iter())
            .map(|normal_tangent_ab|
                normal_tangent_ab.iter()
                .map(|normal_tangent_ab_m|
                    normal_tangent_ab_m.iter()
                    .map(|normal_tangent_ab_mn|
                        (normal_tangent_ab_mn * &traction) * self.get_integration_weight()
                    ).collect()
                ).collect()
            ).collect()
        ).collect::<NodalStiffnesses<N>>())
    }
}

impl<'a, C> ViscoelasticFiniteElement<'a, C, G, N> for Wedge<C>
where
    C: Viscoelastic<'a>,
{
    fn calculate_nodal_forces(
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
        let normal_gradients =
            Self::calculate_normal_gradients(&Self::calculate_midplane(nodal_coordinates));
        let traction = (&first_piola_kirchoff_stress * self.get_reference_normal()) * 0.5;
        Ok(self
            .get_gradient_vectors()
            .iter()
            .zip(normal_gradients.iter().chain(normal_gradients.iter()))
            .map(|(gradient_vector_a, normal_gradient_a)| {
                (&first_piola_kirchoff_stress * gradient_vector_a + normal_gradient_a * &traction)
                    * self.get_integration_weight()
            })
            .collect())
    }
    fn calculate_nodal_stiffnesses(
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
        let normal_gradients =
            Self::calculate_normal_gradients(&Self::calculate_midplane(nodal_coordinates));
        let reference_normal = self.get_reference_normal() * 0.5;
        Ok(gradient_vectors.iter()
        .zip(normal_gradients.iter()
        .chain(normal_gradients.iter()))
        .map(|(gradient_vector_a, normal_gradient_a)|
            gradient_vectors.iter()
            .zip(normal_gradients.iter()
            .chain(normal_gradients.iter()))
            .map(|(gradient_vector_b, normal_gradient_b)|
                IDENTITY.iter()
                .zip(normal_gradient_a.iter())
                .map(|(identity_m, normal_gradient_a_m)|
                    IDENTITY.iter()
                    .zip(normal_gradient_b.iter())
                    .map(|(identity_n, normal_gradient_b_n)|
                        first_piola_kirchoff_tangent_stiffness.iter()
                        .zip(identity_m.iter()
                        .zip(normal_gradient_a_m.iter()))
                        .map(|(first_piola_kirchoff_tangent_stiffness_i, (identity_mi, normal_gradient_a_m_i))|
                            first_piola_kirchoff_tangent_stiffness_i.iter()
                            .zip(gradient_vector_a.iter()
                            .zip(reference_normal.iter()))
                            .map(|(first_piola_kirchoff_tangent_stiffness_ij, (gradient_vector_a_j, reference_normal_j))|
                                first_piola_kirchoff_tangent_stiffness_ij.iter()
                                .zip(identity_n.iter()
                                .zip(normal_gradient_b_n.iter()))
                                .map(|(first_piola_kirchoff_tangent_stiffness_ijk, (identity_nk, normal_gradient_b_n_k))|
                                    first_piola_kirchoff_tangent_stiffness_ijk.iter()
                                    .zip(gradient_vector_b.iter()
                                    .zip(reference_normal.iter()))
                                    .map(|(first_piola_kirchoff_tangent_stiffness_ijkl, (gradient_vector_b_l, reference_normal_l))|
                                        first_piola_kirchoff_tangent_stiffness_ijkl * (
                                            identity_mi * gradient_vector_a_j + normal_gradient_a_m_i * reference_normal_j
                                        ) * (
                                            identity_nk * gradient_vector_b_l + normal_gradient_b_n_k * reference_normal_l
                                        ) * self.get_integration_weight()
                                    ).sum::<Scalar>()
                                ).sum::<Scalar>()
                            ).sum::<Scalar>()
                        ).sum::<Scalar>()
                    ).collect()
                ).collect()
            ).collect()
        ).collect())
    }
}

impl<'a, C> LinearSurfaceElement<'a, C, G, M, N, O> for Wedge<C>
where
    C: Constitutive<'a>,
{
    fn get_reference_normal(&self) -> &ReferenceNormal {
        &self.reference_normal
    }
}

impl<'a, C> ElasticLinearElement<'a, C, G, M, N, O> for Wedge<C> where C: Elastic<'a> {}

impl<'a, C> HyperelasticFiniteElement<'a, C, G, N> for Wedge<C>
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

impl<'a, C> HyperelasticLinearElement<'a, C, G, M, N, O> for Wedge<C> where C: Hyperelastic<'a> {}

impl<'a, C> ViscoelasticLinearElement<'a, C, G, M, N, O> for Wedge<C> where C: Viscoelastic<'a> {}

impl<'a, C> ElasticHyperviscousFiniteElement<'a, C, G, N> for Wedge<C>
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

impl<'a, C> ElasticHyperviscousLinearElement<'a, C, G, M, N, O> for Wedge<C> where
    C: ElasticHyperviscous<'a>
{
}

impl<'a, C> HyperviscoelasticFiniteElement<'a, C, G, N> for Wedge<C>
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

impl<'a, C> HyperviscoelasticLinearElement<'a, C, G, M, N, O> for Wedge<C> where
    C: Hyperviscoelastic<'a>
{
}
