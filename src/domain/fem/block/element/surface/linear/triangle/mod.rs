#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{
            elastic::Elastic, elastic_hyperviscous::ElasticHyperviscous,
            hyperelastic::Hyperelastic, hyperviscoelastic::Hyperviscoelastic,
            viscoelastic::Viscoelastic,
        },
    },
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
        FiniteElement, FiniteElementError, ParametricCoordinate, ParametricCoordinates,
        ShapeFunctions, ShapeFunctionsGradients,
        solid::{
            ElementNodalForcesSolid, ElementNodalStiffnessesSolid, SolidFiniteElement,
            elastic::ElasticFiniteElement, elastic_hyperviscous::ElasticHyperviscousFiniteElement,
            hyperelastic::HyperelasticFiniteElement,
            hyperviscoelastic::HyperviscoelasticFiniteElement,
            viscoelastic::ViscoelasticFiniteElement,
        },
        surface::{SurfaceElement, SurfaceFiniteElement, SurfaceFiniteElementMethods},
    },
    math::{IDENTITY, Scalar, ScalarList, Tensor},
    mechanics::{
        DeformationGradient, DeformationGradientList, DeformationGradientRate,
        DeformationGradientRateList, FirstPiolaKirchhoffRateTangentStiffnesses,
        FirstPiolaKirchhoffStressList, FirstPiolaKirchhoffTangentStiffnessList,
    },
};

const G: usize = 1;
const M: usize = 2;
const N: usize = 3;
const P: usize = G;

pub type Triangle = SurfaceElement<G, N, P>;

impl FiniteElement<G, M, N> for Triangle {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [[0.25; M]].into()
    }
    fn parametric_reference() -> ElementNodalReferenceCoordinates<N> {
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 2.0; G].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2] = parametric_coordinate.into();
        [1.0 - xi_1 - xi_2, xi_1, xi_2].into()
    }
    fn shape_functions_gradients(
        _parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]].into()
    }
}

impl SurfaceFiniteElement<G, N, P> for Triangle {
    fn new(
        reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>,
        thickness: Scalar,
    ) -> Self {
        let integration_weights = Self::bases(&reference_nodal_coordinates)
            .iter()
            .zip(Self::parametric_weights())
            .map(|(reference_basis, parametric_weight)| {
                reference_basis[0].cross(&reference_basis[1]).norm() * parametric_weight * thickness
            })
            .collect();
        let reference_dual_bases = Self::dual_bases(&reference_nodal_coordinates);
        let gradient_vectors = Self::shape_functions_gradients_at_integration_points()
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
            gradient_vectors,
            integration_weights,
            reference_normals,
        }
    }
}

impl SolidFiniteElement<G, N> for Triangle {
    fn deformation_gradients(
        &self,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> DeformationGradientList<G> {
        self.gradient_vectors()
            .iter()
            .zip(
                Self::normals(nodal_coordinates)
                    .iter()
                    .zip(self.reference_normals()),
            )
            .map(|(gradient_vectors, normal_and_reference_normal)| {
                nodal_coordinates
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_coordinate, gradient_vector)| {
                        (nodal_coordinate, gradient_vector).into()
                    })
                    .sum::<DeformationGradient>()
                    + DeformationGradient::from(normal_and_reference_normal)
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> DeformationGradientRateList<G> {
        self.gradient_vectors()
            .iter()
            .zip(
                Self::normal_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.reference_normals()),
            )
            .map(|(gradient_vectors, normal_rate_and_reference_normal)| {
                nodal_velocities
                    .iter()
                    .zip(gradient_vectors.iter())
                    .map(|(nodal_velocity, gradient_vector)| {
                        (nodal_velocity, gradient_vector).into()
                    })
                    .sum::<DeformationGradientRate>()
                    + DeformationGradientRate::from(normal_rate_and_reference_normal)
            })
            .collect()
    }
}

impl<C> ElasticFiniteElement<C, G, N> for Triangle
where
    C: Elastic,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_stress(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffStressList<G>, _>>()
        {
            Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
                .iter()
                .zip(
                    self.gradient_vectors()
                        .iter()
                        .zip(self.integration_weights()),
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
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        match self.deformation_gradients(nodal_coordinates).iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnessList<G>, _>>()
            {
            Ok(first_piola_kirchhoff_tangent_stiffnesses) => Ok(first_piola_kirchhoff_tangent_stiffnesses
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights().iter()
                    .zip(self.reference_normals().iter()
                    .zip(Self::normal_gradients(nodal_coordinates))
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

impl<C> HyperelasticFiniteElement<C, G, N> for Triangle
where
    C: Hyperelastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(self.integration_weights())
            .map(|(deformation_gradient, integration_weight)| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                        * integration_weight,
                )
            })
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

impl<C> ViscoelasticFiniteElement<C, G, N> for Triangle
where
    C: Viscoelastic,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter(),
            )
            .map(|(deformation_gradient, deformation_gradient_rate)| {
                constitutive_model
                    .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_rate)
            })
            .collect::<Result<FirstPiolaKirchhoffStressList<G>, _>>()
        {
            Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
                .iter()
                .zip(
                    self.gradient_vectors()
                        .iter()
                        .zip(self.integration_weights()),
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
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        match self.deformation_gradients(nodal_coordinates).iter().zip(self.deformation_gradient_rates(nodal_coordinates, nodal_velocities).iter())
            .map(|(deformation_gradient, deformation_gradient_rate)| {
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
                    .zip(Self::normal_gradients(nodal_coordinates))
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

impl<C> ElasticHyperviscousFiniteElement<C, G, N> for Triangle
where
    C: ElasticHyperviscous,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
                    Ok::<_, ConstitutiveError>(
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
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
                    Ok::<_, ConstitutiveError>(
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

impl<C> HyperviscoelasticFiniteElement<C, G, N> for Triangle
where
    C: Hyperviscoelastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(self.integration_weights())
            .map(|(deformation_gradient, integration_weight)| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                        * integration_weight,
                )
            })
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
