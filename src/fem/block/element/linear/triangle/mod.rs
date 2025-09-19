#[cfg(test)]
mod test;

use super::*;
use crate::{
    constitutive::{Constitutive, Parameters},
    math::{IDENTITY, tensor_rank_1, tensor_rank_1_list, tensor_rank_1_list_2d},
    mechanics::Scalar,
};
use std::array::from_fn;

const G: usize = 1;
const M: usize = 2;
const N: usize = 3;
const P: usize = G;

#[cfg(test)]
const Q: usize = N;

pub type Triangle<C> = SurfaceElement<C, G, N, P>;

impl<C, Y> SurfaceFiniteElement<C, G, N, P, Y> for Triangle<C>
where
    C: Constitutive<Y>,
    Y: Parameters,
{
    fn new(
        constitutive_model_parameters: Y,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
        thickness: &Scalar,
    ) -> Self {
        let integration_weights = Self::bases(&reference_nodal_coordinates)
            .iter()
            .map(|reference_basis| {
                reference_basis[0].cross(&reference_basis[1]).norm()
                    * Self::integration_weight()
                    * thickness
            })
            .collect();
        let reference_dual_bases = Self::dual_bases(&reference_nodal_coordinates);
        let gradient_vectors = Self::standard_gradient_operators()
            .iter()
            .zip(reference_dual_bases.iter())
            .map(|(standard_gradient_operator, reference_dual_basis)| {
                standard_gradient_operator
                    .iter()
                    .map(|standard_gradient_operator_a| {
                        standard_gradient_operator_a
                            .iter()
                            .zip(reference_dual_basis.iter())
                            .map(|(standard_gradient_operator_a_m, reference_dual_basis_m)| {
                                reference_dual_basis_m * standard_gradient_operator_a_m
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect();
        let reference_normals = reference_dual_bases
            .iter()
            .map(|reference_dual_basis| {
                reference_dual_basis[0]
                    .cross(&reference_dual_basis[1])
                    .normalized()
            })
            .collect();
        Self {
            constitutive_models: from_fn(|_| <C>::new(constitutive_model_parameters)),
            gradient_vectors,
            integration_weights,
            reference_normals,
        }
    }
}

impl<C> Triangle<C> {
    const fn integration_weight() -> Scalar {
        1.0 / 2.0
    }
    #[cfg(test)]
    const fn shape_functions_at_integration_points() -> ShapeFunctionsAtIntegrationPoints<G, Q> {
        tensor_rank_1_list([tensor_rank_1([1.0 / 3.0; Q])])
    }
    const fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        tensor_rank_1_list_2d([tensor_rank_1_list([
            tensor_rank_1([-1.0, -1.0]),
            tensor_rank_1([1.0, 0.0]),
            tensor_rank_1([0.0, 1.0]),
        ])])
    }
}

impl<C> SurfaceFiniteElementMethodsExtra<M, N, P> for Triangle<C> {
    fn standard_gradient_operators() -> StandardGradientOperators<M, N, P> {
        Self::standard_gradient_operators()
    }
}

impl<C> FiniteElementMethods<C, G, N> for Triangle<C> {
    fn constitutive_models(&self) -> &[C; G] {
        &self.constitutive_models
    }
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradientList<G> {
        self.gradient_vectors()
            .iter()
            .zip(
                Self::normals(nodal_coordinates)
                    .iter()
                    .zip(self.reference_normals().iter()),
            )
            .map(|(gradient_vectors, (normal, reference_normal))| {
                nodal_coordinates
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_coordinate, gradient_vector)| {
                        DeformationGradient::dyad(nodal_coordinate, gradient_vector)
                    })
                    .sum::<DeformationGradient>()
                    + DeformationGradient::dyad(normal, reference_normal)
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRateList<G> {
        self.gradient_vectors()
            .iter()
            .zip(
                Self::normal_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.reference_normals().iter()),
            )
            .map(|(gradient_vectors, (normal_rate, reference_normal))| {
                nodal_velocities
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_velocity, gradient_vector)| {
                        DeformationGradientRate::dyad(nodal_velocity, gradient_vector)
                    })
                    .sum::<DeformationGradientRate>()
                    + DeformationGradientRate::dyad(normal_rate, reference_normal)
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

impl<C> ElasticFiniteElement<C, G, N> for Triangle<C>
where
    C: Elastic,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, FiniteElementError> {
        match self
            .constitutive_models()
            .iter()
            .zip(self.deformation_gradients(nodal_coordinates).iter())
            .map(|(constitutive_model, deformation_gradient)| {
                constitutive_model.first_piola_kirchhoff_stress(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffStresses<G>, _>>()
        {
            Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
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
                                (first_piola_kirchhoff_stress * gradient_vector)
                                    * integration_weight
                            })
                            .collect()
                    },
                )
                .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError> {
        match self
            .constitutive_models()
            .iter()
            .zip(self.deformation_gradients(nodal_coordinates).iter())
            .map(|(constitutive_model, deformation_gradient)| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses<G>, _>>()
            {
            Ok(first_piola_kirchhoff_tangent_stiffnesses) => Ok(first_piola_kirchhoff_tangent_stiffnesses
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights().iter()
                    .zip(self.reference_normals().iter()
                    .zip(Self::normal_gradients(nodal_coordinates).iter())
                )
                ),
            )
            .map(
                |(
                    first_piola_kirchhoff_tangent_stiffness,
                    (gradient_vectors, (integration_weight, (reference_normal, normal_gradients))),
                )| {
                    gradient_vectors.iter()
                    .map(|gradient_vector_a|
                        gradient_vectors.iter()
                        .zip(normal_gradients.iter())
                        .map(|(gradient_vector_b, normal_gradient_b)|
                            first_piola_kirchhoff_tangent_stiffness.iter()
                            .map(|first_piola_kirchhoff_tangent_stiffness_m|
                                IDENTITY.iter()
                                .zip(normal_gradient_b.iter())
                                .map(|(identity_n, normal_gradient_b_n)|
                                    first_piola_kirchhoff_tangent_stiffness_m.iter()
                                    .zip(gradient_vector_a.iter())
                                    .map(|(first_piola_kirchhoff_tangent_stiffness_mj, gradient_vector_a_j)|
                                        first_piola_kirchhoff_tangent_stiffness_mj.iter()
                                        .zip(identity_n.iter()
                                        .zip(normal_gradient_b_n.iter()))
                                        .map(|(first_piola_kirchhoff_tangent_stiffness_mjk, (identity_nk, normal_gradient_b_n_k))|
                                            first_piola_kirchhoff_tangent_stiffness_mjk.iter()
                                            .zip(gradient_vector_b.iter()
                                            .zip(reference_normal.iter()))
                                            .map(|(first_piola_kirchhoff_tangent_stiffness_mjkl, (gradient_vector_b_l, reference_normal_l))|
                                                first_piola_kirchhoff_tangent_stiffness_mjkl * gradient_vector_a_j * (
                                                    identity_nk * gradient_vector_b_l + normal_gradient_b_n_k * reference_normal_l
                                                ) * integration_weight
                                            ).sum::<Scalar>()
                                        ).sum::<Scalar>()
                                    ).sum::<Scalar>()
                                ).collect()
                            ).collect()
                        ).collect()
                    ).collect()
                }
            )
            .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C> HyperelasticFiniteElement<C, G, N> for Triangle<C>
where
    C: Hyperelastic,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates)
                    .iter()
                    .zip(self.integration_weights().iter()),
            )
            .map(
                |(constitutive_model, (deformation_gradient, integration_weight))| {
                    Ok::<Scalar, ConstitutiveError>(
                        constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                            * integration_weight,
                    )
                },
            )
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C> ViscoelasticFiniteElement<C, G, N> for Triangle<C>
where
    C: Viscoelastic,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, FiniteElementError> {
        match self
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
            .collect::<Result<FirstPiolaKirchhoffStresses<G>, _>>()
        {
            Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
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
                                (first_piola_kirchhoff_stress * gradient_vector)
                                    * integration_weight
                            })
                            .collect()
                    },
                )
                .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError> {
        match self
            .constitutive_models()
            .iter()
            .zip(self.deformation_gradients(nodal_coordinates).iter().zip(self.deformation_gradient_rates(nodal_coordinates, nodal_velocities).iter()))
            .map(|(constitutive_model, (deformation_gradient, deformation_gradient_rate))| {
                constitutive_model.first_piola_kirchhoff_rate_tangent_stiffness(deformation_gradient, deformation_gradient_rate)
            })
            .collect::<Result<FirstPiolaKirchhoffRateTangentStiffnesses<G>, _>>()
        {
            Ok(first_piola_kirchhoff_rate_tangent_stiffnesses) => Ok(first_piola_kirchhoff_rate_tangent_stiffnesses
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights().iter()
                    .zip(self.reference_normals().iter()
                    .zip(Self::normal_gradients(nodal_coordinates).iter())
                )
                ),
            )
            .map(
                |(
                    first_piola_kirchoff_rate_tangent_stiffness_mjkl,
                    (gradient_vectors, (integration_weight, (reference_normal, normal_gradients))),
                )| {
                    gradient_vectors.iter()
                    .map(|gradient_vector_a|
                        gradient_vectors.iter()
                        .zip(normal_gradients.iter())
                        .map(|(gradient_vector_b, normal_gradient_b)|
                            first_piola_kirchoff_rate_tangent_stiffness_mjkl.iter()
                            .map(|first_piola_kirchhoff_rate_tangent_stiffness_m|
                                IDENTITY.iter()
                                .zip(normal_gradient_b.iter())
                                .map(|(identity_n, normal_gradient_b_n)|
                                    first_piola_kirchhoff_rate_tangent_stiffness_m.iter()
                                    .zip(gradient_vector_a.iter())
                                    .map(|(first_piola_kirchhoff_rate_tangent_stiffness_mj, gradient_vector_a_j)|
                                        first_piola_kirchhoff_rate_tangent_stiffness_mj.iter()
                                        .zip(identity_n.iter()
                                        .zip(normal_gradient_b_n.iter()))
                                        .map(|(first_piola_kirchhoff_rate_tangent_stiffness_mjk, (identity_nk, normal_gradient_b_n_k))|
                                            first_piola_kirchhoff_rate_tangent_stiffness_mjk.iter()
                                            .zip(gradient_vector_b.iter()
                                            .zip(reference_normal.iter()))
                                            .map(|(first_piola_kirchoff_rate_tangent_stiffness_mjkl, (gradient_vector_b_l, reference_normal_l))|
                                                first_piola_kirchoff_rate_tangent_stiffness_mjkl * gradient_vector_a_j * (
                                                    identity_nk * gradient_vector_b_l + normal_gradient_b_n_k * reference_normal_l
                                                ) * integration_weight
                                            ).sum::<Scalar>()
                                        ).sum::<Scalar>()
                                    ).sum::<Scalar>()
                                ).collect()
                            ).collect()
                        ).collect()
                    ).collect()
                }
            )
            .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C> ElasticHyperviscousFiniteElement<C, G, N> for Triangle<C>
where
    C: ElasticHyperviscous,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .constitutive_models()
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
                    Ok::<Scalar, ConstitutiveError>(
                        constitutive_model
                            .viscous_dissipation(deformation_gradient, deformation_gradient_rate)?
                            * integration_weight,
                    )
                },
            )
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .constitutive_models()
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
                    Ok::<Scalar, ConstitutiveError>(
                        constitutive_model.dissipation_potential(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )? * integration_weight,
                    )
                },
            )
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C> HyperviscoelasticFiniteElement<C, G, N> for Triangle<C>
where
    C: Hyperviscoelastic,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .constitutive_models()
            .iter()
            .zip(
                self.deformation_gradients(nodal_coordinates)
                    .iter()
                    .zip(self.integration_weights().iter()),
            )
            .map(
                |(constitutive_model, (deformation_gradient, integration_weight))| {
                    Ok::<Scalar, ConstitutiveError>(
                        constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                            * integration_weight,
                    )
                },
            )
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
