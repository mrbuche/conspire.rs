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

pub trait SurfaceElementMethods<'a, C, const G: usize, const N: usize, const P: usize>
where
    C: Constitutive<'a>,
    Self: FiniteElementMethods<'a, C, G, N>
{
    fn bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P>;
    fn dual_bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P>;
    fn normals(nodal_coordinates: &NodalCoordinates<N>) -> Normals<P>;
    fn normal_rates(nodal_coordinates: &NodalCoordinates<N>, nodal_velocities: &NodalVelocities<N>) -> NormalRates<P>;
    fn reference_normals(&self) -> &ReferenceNormals<P>;
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

pub trait SurfaceFiniteElement<'a, C, const G: usize, const N: usize>
where
    C: Constitutive<'a>,
    Self: SurfaceElementMethods<'a, C, G, N>,
{
    fn new(
        constitutive_model_parameters: Parameters<'a>,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
        thickness: &Scalar,
    ) -> Self;
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

impl<'a, C, const G: usize, const N: usize>
    FiniteElementMethods<'a, C, G, N> for SurfaceElement<C, G, N>
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
            .zip(Self::normals(nodal_coordinates).iter().zip(self.reference_normals().iter()))
            .map(|(gradient_vectors, (normal, reference_normal))| {
                nodal_coordinates
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_coordinate, gradient_vector)| {
                        DeformationGradient::dyad(nodal_coordinate, gradient_vector)
                    })
                    .sum::<DeformationGradient>()
                    + DeformationGradient::dyad(
                        normal,
                        reference_normal,
                    )
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRates<G> {
        self.gradient_vectors()
            .iter()
            .zip(Self::normal_rates(nodal_coordinates, nodal_velocities).iter().zip(self.reference_normals().iter()))
            .map(|(gradient_vectors, (normal_rate, reference_normal))| {
                nodal_velocities
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_velocity, gradient_vector)| {
                        DeformationGradientRate::dyad(nodal_velocity, gradient_vector)
                    })
                    .sum::<DeformationGradientRate>()
                    + DeformationGradientRate::dyad(
                        normal_rate,
                        reference_normal,
                    )
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

impl<'a, C, const G: usize, const N: usize, const P: usize>
    SurfaceElementMethods<'a, C, G, N, P> for SurfaceElement<C, G, N>
where
    C: Constitutive<'a>,
{
    fn bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P> {
        todo!()
        // Self::standard_gradient_operators()
        //     .iter()
        //     .map(|standard_gradient_operator| {
        //         standard_gradient_operator
        //             .iter()
        //             .zip(nodal_coordinates.iter())
        //             .map(|(standard_gradient_operator_a, nodal_coordinate_a)| {
        //                 standard_gradient_operator_a
        //                     .iter()
        //                     .map(|standard_gradient_operator_a_m| {
        //                         nodal_coordinate_a * standard_gradient_operator_a_m
        //                     })
        //                     .collect()
        //             })
        //             .sum()
        //     })
        //     .collect()
    }
    fn dual_bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P> {
        todo!()
        // Self::bases(nodal_coordinates)
        //     .iter()
        //     .map(|basis_vectors| {
        //         basis_vectors
        //             .iter()
        //             .map(|basis_vectors_m| {
        //                 basis_vectors
        //                     .iter()
        //                     .map(|basis_vectors_n| basis_vectors_m * basis_vectors_n)
        //                     .collect()
        //             })
        //             .collect::<TensorRank2<M, I, I>>()
        //             .inverse()
        //             .iter()
        //             .map(|metric_tensor_m| {
        //                 metric_tensor_m
        //                     .iter()
        //                     .zip(basis_vectors.iter())
        //                     .map(|(metric_tensor_mn, basis_vectors_n)| {
        //                         basis_vectors_n * metric_tensor_mn
        //                     })
        //                     .sum()
        //             })
        //             .collect()
        //     })
        //     .collect()
    }
    fn normals(nodal_coordinates: &NodalCoordinates<N>) -> Normals<P> {
        Self::bases(nodal_coordinates)
            .iter()
            .map(|basis_vectors| basis_vectors[0].cross(&basis_vectors[1]).normalized())
            .collect()
    }
    fn normal_rates(nodal_coordinates: &NodalCoordinates<N>, nodal_velocities: &NodalVelocities<N>) -> NormalRates<P> {
        todo!()
    }
    fn reference_normals(&self) -> &ReferenceNormals<P> {
        &self.reference_normals
    }
}

pub trait ElasticFiniteElement<'a, C, const N: usize>
where
    C: Elastic<'a>,
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

pub trait HyperelasticFiniteElement<'a, C, const N: usize>
where
    C: Hyperelastic<'a>,
    Self: ElasticFiniteElement<'a, C, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError>;
}

pub trait ViscoelasticFiniteElement<'a, C, const N: usize>
where
    C: Viscoelastic<'a>,
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

pub trait ElasticHyperviscousFiniteElement<'a, C, const N: usize>
where
    C: ElasticHyperviscous<'a>,
    Self: ViscoelasticFiniteElement<'a, C, N>,
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

pub trait HyperviscoelasticFiniteElement<'a, C, const N: usize>
where
    C: Hyperviscoelastic<'a>,
    Self: ElasticHyperviscousFiniteElement<'a, C, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError>;
}

impl<'a, C, const G: usize, const N: usize>
    ElasticFiniteElement<'a, C, N> for Element<C, G, N>
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
    HyperelasticFiniteElement<'a, C, N> for Element<C, G, N>
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
    ViscoelasticFiniteElement<'a, C, N> for Element<C, G, N>
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
    ElasticHyperviscousFiniteElement<'a, C, N> for Element<C, G, N>
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
    HyperviscoelasticFiniteElement<'a, C, N> for Element<C, G, N>
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
