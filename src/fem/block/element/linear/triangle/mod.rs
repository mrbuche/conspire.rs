#[cfg(test)]
mod test;

use super::*;
use crate::{
    constitutive::{Constitutive, Parameters},
    math::{
        IDENTITY, LEVI_CIVITA, tensor_rank_1, tensor_rank_1_list, tensor_rank_1_list_2d,
        tensor_rank_1_zero,
    },
    mechanics::Scalar,
};
use std::array::from_fn;

const G: usize = 1;
const M: usize = 2;
const N: usize = 3;
const P: usize = 1;

#[cfg(test)]
const Q: usize = 3;

pub type Triangle<C> = SurfaceElement<C, G, N, P>;

impl<C, Y> SurfaceFiniteElement<C, G, N, P, Y> for Triangle<C>
where
    C: Constitutive<Y>,
    Y: Parameters,
{
    fn bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P> {
        Self::standard_gradient_operators()
            .iter()
            .map(|standard_gradient_operator| {
                standard_gradient_operator
                    .iter()
                    .zip(nodal_coordinates.iter())
                    .map(|(standard_gradient_operator_a, nodal_coordinate_a)| {
                        standard_gradient_operator_a
                            .iter()
                            .map(|standard_gradient_operator_a_m| {
                                nodal_coordinate_a * standard_gradient_operator_a_m
                            })
                            .collect()
                    })
                    .sum()
            })
            .collect()
    }
    fn dual_bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P> {
        Self::bases(nodal_coordinates)
            .iter()
            .map(|basis_vectors| {
                basis_vectors
                    .iter()
                    .map(|basis_vectors_m| {
                        basis_vectors
                            .iter()
                            .map(|basis_vectors_n| basis_vectors_m * basis_vectors_n)
                            .collect()
                    })
                    .collect::<TensorRank2<2, I, I>>()
                    .inverse()
                    .iter()
                    .map(|metric_tensor_m| {
                        metric_tensor_m
                            .iter()
                            .zip(basis_vectors.iter())
                            .map(|(metric_tensor_mn, basis_vectors_n)| {
                                basis_vectors_n * metric_tensor_mn
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
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
    fn normals(nodal_coordinates: &NodalCoordinates<N>) -> Normals<P> {
        Self::bases(nodal_coordinates)
            .iter()
            .map(|basis_vectors| basis_vectors[0].cross(&basis_vectors[1]).normalized())
            .collect()
    }
    fn normal_gradients(nodal_coordinates: &NodalCoordinates<N>) -> NormalGradients<N, P> {
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization: Scalar = 0.0;
        let mut normal_vector = tensor_rank_1_zero();
        Self::standard_gradient_operators().iter()
        .zip(Self::bases(nodal_coordinates).iter())
        .map(|(standard_gradient_operator, basis_vectors)|{
            normalization = basis_vectors[0].cross(&basis_vectors[1]).norm();
            normal_vector = basis_vectors[0].cross(&basis_vectors[1])/normalization;
            standard_gradient_operator.iter()
            .map(|standard_gradient_operator_a|
                levi_civita_symbol.iter()
                .map(|levi_civita_symbol_m|
                    IDENTITY.iter()
                    .zip(normal_vector.iter())
                    .map(|(identity_i, normal_vector_i)|
                        levi_civita_symbol_m.iter()
                        .zip(basis_vectors[0].iter()
                        .zip(basis_vectors[1].iter()))
                        .map(|(levi_civita_symbol_mn, (basis_vector_0_n, basis_vector_1_n))|
                            levi_civita_symbol_mn.iter()
                            .zip(identity_i.iter()
                            .zip(normal_vector.iter()))
                            .map(|(levi_civita_symbol_mno, (identity_io, normal_vector_o))|
                                levi_civita_symbol_mno * (identity_io - normal_vector_i * normal_vector_o)
                            ).sum::<Scalar>() * (
                                standard_gradient_operator_a[0] * basis_vector_1_n
                              - standard_gradient_operator_a[1] * basis_vector_0_n
                            )
                        ).sum::<Scalar>() / normalization
                    ).collect()
                ).collect()
            ).collect()
        }).collect()
    }
    fn normal_rates(
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> NormalRates<P> {
        let identity = IDENTITY;
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization = 0.0;
        Self::bases(nodal_coordinates)
            .iter()
            .zip(Self::normals(nodal_coordinates).iter()
            .zip(Self::standard_gradient_operators().iter()))
            .map(|(basis, (normal, standard_gradient_operator))| {
                normalization = basis[0].cross(&basis[1]).norm();
                identity.iter()
                .zip(normal.iter())
                .map(|(identity_i, normal_vector_i)|
                    nodal_velocities.iter()
                    .zip(standard_gradient_operator.iter())
                    .map(|(nodal_velocity_a, standard_gradient_operator_a)|
                        levi_civita_symbol.iter()
                        .zip(nodal_velocity_a.iter())
                        .map(|(levi_civita_symbol_m, nodal_velocity_a_m)|
                            levi_civita_symbol_m.iter()
                            .zip(basis[0].iter()
                            .zip(basis[1].iter()))
                            .map(|(levi_civita_symbol_mn, (basis_vector_0_n, basis_vector_1_n))|
                                levi_civita_symbol_mn.iter()
                                .zip(identity_i.iter()
                                .zip(normal.iter()))
                                .map(|(levi_civita_symbol_mno, (identity_io, normal_vector_o))|
                                    levi_civita_symbol_mno * (identity_io - normal_vector_i * normal_vector_o)
                                ).sum::<Scalar>() * (
                                    standard_gradient_operator_a[0] * basis_vector_1_n
                                - standard_gradient_operator_a[1] * basis_vector_0_n
                                )
                            ).sum::<Scalar>() * nodal_velocity_a_m
                        ).sum::<Scalar>()
                    ).sum::<Scalar>() / normalization
                ).collect()
        }).collect()
    }
    fn reference_normals(&self) -> &ReferenceNormals<P> {
        &self.reference_normals
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

impl<C, Y> FiniteElementMethods<C, G, N, Y> for Triangle<C>
where
    C: Constitutive<Y>,
    Y: Parameters,
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
    ) -> DeformationGradientRates<G> {
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

impl<C, Y> ElasticFiniteElement<C, G, N, Y> for Triangle<C>
where
    C: Elastic<Y>,
    Y: Parameters,
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
            .sum())
    }
}

impl<C, Y> HyperelasticFiniteElement<C, G, N, Y> for Triangle<C>
where
    C: Hyperelastic<Y>,
    Y: Parameters,
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

impl<C, Y> ViscoelasticFiniteElement<C, G, N, Y> for Triangle<C>
where
    C: Viscoelastic<Y>,
    Y: Parameters,
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
            .zip(self.deformation_gradients(nodal_coordinates).iter().zip(self.deformation_gradient_rates(nodal_coordinates, nodal_velocities).iter()))
            .map(|(constitutive_model, (deformation_gradient, deformation_gradient_rate))| {
                constitutive_model.first_piola_kirchhoff_rate_tangent_stiffness(deformation_gradient, deformation_gradient_rate)
            })
            .collect::<Result<FirstPiolaKirchhoffRateTangentStiffnesses<G>, _>>()?
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
            .sum())
    }
}

impl<C, Y> ElasticHyperviscousFiniteElement<C, G, N, Y> for Triangle<C>
where
    C: ElasticHyperviscous<Y>,
    Y: Parameters,
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

impl<C, Y> HyperviscoelasticFiniteElement<C, G, N, Y> for Triangle<C>
where
    C: Hyperviscoelastic<Y>,
    Y: Parameters,
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
