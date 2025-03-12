#[cfg(test)]
mod test;

pub mod composite;
pub mod linear;

use super::*;

pub struct Element<C, const G: usize, const N: usize> {
    constitutive_models: [C; G],
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars<G>,
}

pub struct SurfaceElement<C, const G: usize, const N: usize, const P: usize> {
    constitutive_models: [C; G],
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars<G>,
    reference_normals: ReferenceNormals<P>,
}

pub trait FiniteElementMethods<'a, C, const G: usize, const N: usize>
where
    C: Constitutive<'a>,
{
    fn constitutive_models(&self) -> &[C; G];
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradients<G>;
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRates<G>;
    fn gradient_vectors(&self) -> &GradientVectors<G, N>;
    fn integration_weights(&self) -> &Scalars<G>;
}

pub trait FiniteElement<'a, C, const G: usize, const N: usize>
where
    C: Constitutive<'a>,
    Self: FiniteElementMethods<'a, C, G, N>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self;
}

pub trait SurfaceFiniteElement<'a, C, const G: usize, const N: usize, const P: usize>
where
    C: Constitutive<'a>,
{
    fn bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P>;
    fn dual_bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P>;
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
        thickness: &Scalar,
    ) -> Self;
    fn normals(nodal_coordinates: &NodalCoordinates<N>) -> Normals<P>;
    fn normal_gradients(nodal_coordinates: &NodalCoordinates<N>) -> NormalGradients<N, P>;
    fn normal_rates(nodal_coordinates: &NodalCoordinates<N>, nodal_velocities: &NodalVelocities<N>) -> NormalRates<P>;
    fn normal_tangents(nodal_coordinates: &NodalCoordinates<N>) -> NormalTangents<N, P>;
    fn reference_normals(&self) -> &ReferenceNormals<P>;
}

impl<'a, C, const G: usize, const N: usize>
    FiniteElementMethods<'a, C, G, N> for Element<C, G, N>
where
    C: Constitutive<'a>,
{
    fn constitutive_models(&self) -> &[C; G] {
        &self.constitutive_models
    }
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradients<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_coordinates
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_coordinate, gradient_vector)| {
                        DeformationGradient::dyad(nodal_coordinate, gradient_vector)
                    })
                    .sum()
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        _: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRates<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_velocities
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_velocity, gradient_vector)| {
                        DeformationGradientRate::dyad(nodal_velocity, gradient_vector)
                    })
                    .sum()
            })
            .collect()
    }
    fn gradient_vectors(&self) -> &GradientVectors<G, N> {
        &self.gradient_vectors
    }
    fn integration_weights(&self) -> &Scalars<G> {
        &self.integration_weights
    }
}

pub trait ElasticFiniteElement<'a, C, const G: usize, const N: usize>
where
    C: Elastic<'a>,
    Self: FiniteElementMethods<'a, C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError>;
}

pub trait HyperelasticFiniteElement<'a, C, const G: usize, const N: usize>
where
    C: Hyperelastic<'a>,
    Self: ElasticFiniteElement<'a, C, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError>;
}

pub trait ViscoelasticFiniteElement<'a, C, const G: usize, const N: usize>
where
    C: Viscoelastic<'a>,
    Self: FiniteElementMethods<'a, C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError>;
}

pub trait ElasticHyperviscousFiniteElement<'a, C, const G: usize, const N: usize>
where
    C: ElasticHyperviscous<'a>,
    Self: ViscoelasticFiniteElement<'a, C, G, N>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError>;
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError>;
}

pub trait HyperviscoelasticFiniteElement<'a, C, const G: usize, const N: usize>
where
    C: Hyperviscoelastic<'a>,
    Self: ElasticHyperviscousFiniteElement<'a, C, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError>;
}

impl<'a, C, const G: usize, const N: usize>
    ElasticFiniteElement<'a, C, G, N> for Element<C, G, N>
where
    C: Elastic<'a>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        Ok(self
            .constitutive_models()
            .iter()
            .zip(self.deformation_gradients(nodal_coordinates).iter())
            .map(|(constitutive_model, deformation_gradient)| {
                constitutive_model.first_piola_kirchhoff_stress(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffStresses<G>, _>>()?
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights().iter()),
            )
            .map(
                |(first_piola_kirchhoff_stress, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector| {
                            (first_piola_kirchhoff_stress * gradient_vector) * integration_weight
                        })
                        .collect()
                },
            )
            .sum())
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        Ok(self
            .constitutive_models()
            .iter()
            .zip(self.deformation_gradients(nodal_coordinates).iter())
            .map(|(constitutive_model, deformation_gradient)| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses<G>, _>>()?
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights().iter()),
            )
            .map(
                |(
                    first_piola_kirchhoff_tangent_stiffness,
                    (gradient_vectors, integration_weight),
                )| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector_a| {
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_b| {
                                    first_piola_kirchhoff_tangent_stiffness
                                        .contract_second_fourth_indices_with_first_indices_of(
                                            gradient_vector_a,
                                            gradient_vector_b,
                                        )
                                        * integration_weight
                                })
                                .collect()
                        })
                        .collect()
                },
            )
            .sum())
    }
}

impl<'a, C, const G: usize, const N: usize>
    HyperelasticFiniteElement<'a, C, G, N> for Element<C, G, N>
where
    C: Hyperelastic<'a>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates)
                    .iter()
                    .zip(self.integration_weights().iter()),
            )
            .map(
                |(constitutive_model, (deformation_gradient, integration_weight))| {
                    Ok(
                        constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                            * integration_weight,
                    )
                },
            )
            .sum()
    }
}

impl<'a, C, const G: usize, const N: usize>
    ViscoelasticFiniteElement<'a, C, G, N> for Element<C, G, N>
where
    C: Viscoelastic<'a>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, ConstitutiveError> {
        Ok(self
            .constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates).iter().zip(
                    self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                        .iter(),
                ),
            )
            .map(
                |(constitutive_model, (deformation_gradient, deformation_gradient_rate))| {
                    constitutive_model.first_piola_kirchhoff_stress(
                        deformation_gradient,
                        deformation_gradient_rate,
                    )
                },
            )
            .collect::<Result<FirstPiolaKirchhoffStresses<G>, _>>()?
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights().iter()),
            )
            .map(
                |(first_piola_kirchhoff_stress, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector| {
                            (first_piola_kirchhoff_stress * gradient_vector) * integration_weight
                        })
                        .collect()
                },
            )
            .sum())
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError> {
        Ok(self
            .constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates).iter().zip(
                    self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                        .iter(),
                ),
            )
            .map(
                |(constitutive_model, (deformation_gradient, deformation_gradient_rate))| {
                    constitutive_model.first_piola_kirchhoff_rate_tangent_stiffness(
                        deformation_gradient,
                        deformation_gradient_rate,
                    )
                },
            )
            .collect::<Result<FirstPiolaKirchhoffRateTangentStiffnesses<G>, _>>()?
            .iter()
            .zip(
                self.gradient_vectors().iter().zip(
                    self.gradient_vectors()
                        .iter()
                        .zip(self.integration_weights().iter()),
                ),
            )
            .map(
                |(
                    first_piola_kirchhoff_rate_tangent_stiffness,
                    (gradient_vectors_a, (gradient_vectors_b, integration_weight)),
                )| {
                    gradient_vectors_a
                        .iter()
                        .map(|gradient_vector_a| {
                            gradient_vectors_b
                                .iter()
                                .map(|gradient_vector_b| {
                                    first_piola_kirchhoff_rate_tangent_stiffness
                                        .contract_second_fourth_indices_with_first_indices_of(
                                            gradient_vector_a,
                                            gradient_vector_b,
                                        )
                                        * integration_weight
                                })
                                .collect()
                        })
                        .collect()
                },
            )
            .sum())
    }
}

impl<'a, C, const G: usize, const N: usize>
    ElasticHyperviscousFiniteElement<'a, C, G, N> for Element<C, G, N>
where
    C: ElasticHyperviscous<'a>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates).iter().zip(
                    self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                        .iter()
                        .zip(self.integration_weights().iter()),
                ),
            )
            .map(
                |(
                    constitutive_model,
                    (deformation_gradient, (deformation_gradient_rate, integration_weight)),
                )| {
                    Ok(constitutive_model
                        .viscous_dissipation(deformation_gradient, deformation_gradient_rate)?
                        * integration_weight)
                },
            )
            .sum()
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates).iter().zip(
                    self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                        .iter()
                        .zip(self.integration_weights().iter()),
                ),
            )
            .map(
                |(
                    constitutive_model,
                    (deformation_gradient, (deformation_gradient_rate, integration_weight)),
                )| {
                    Ok(constitutive_model
                        .dissipation_potential(deformation_gradient, deformation_gradient_rate)?
                        * integration_weight)
                },
            )
            .sum()
    }
}

impl<'a, C, const G: usize, const N: usize>
    HyperviscoelasticFiniteElement<'a, C, G, N> for Element<C, G, N>
where
    C: Hyperviscoelastic<'a>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError> {
        self.constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates)
                    .iter()
                    .zip(self.integration_weights().iter()),
            )
            .map(
                |(constitutive_model, (deformation_gradient, integration_weight))| {
                    Ok(
                        constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                            * integration_weight,
                    )
                },
            )
            .sum()
    }
}
