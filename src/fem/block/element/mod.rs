#[cfg(test)]
mod test;

pub mod composite;
pub mod linear;

use super::*;
use crate::{
    constitutive::{Constitutive, Parameters},
    defeat_message,
    math::{IDENTITY, LEVI_CIVITA, tensor_rank_1_zero},
    mechanics::Scalar,
};
use std::fmt::{Debug, Display};

// #[derive(Debug)]
pub struct Element<C, const G: usize, const N: usize> {
    constitutive_models: [C; G],
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars<G>,
}

#[derive(Debug)]
pub struct SurfaceElement<C, const G: usize, const N: usize, const P: usize> {
    constitutive_models: [C; G],
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars<G>,
    reference_normals: ReferenceNormals<P>,
}

impl<C, const G: usize, const N: usize> Debug for Element<C, G, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match (G, N) {
            (1, 4) => write!(f, "LinearTetrahedron {{ constitutive_model: {} }}", "Foo"), // get Display implemented for the constitutive models
            (8, 8) => write!(f, "LinearHexahedron {{ constitutive_model: {} }}", "Foo"), // also may want to consider impl debug separately to show parameter names
            (4, 10) => write!(
                f,
                "CompositeTetrahedron {{ constitutive_model: {} }}",
                "Foo"
            ),
            _ => panic!(),
        }
    }
}

pub trait FiniteElement<C, const G: usize, const N: usize, Y>
where
    C: Constitutive<Y>,
    Self: FiniteElementMethods<C, G, N>,
    Y: Parameters,
{
    fn new(
        constitutive_model_parameters: Y,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
    ) -> Self;
    fn reference() -> ReferenceNodalCoordinates<N>;
    fn reset(&mut self);
}

pub trait SurfaceFiniteElement<C, const G: usize, const N: usize, const P: usize, Y>
where
    C: Constitutive<Y>,
    Self: FiniteElementMethods<C, G, N>,
    Y: Parameters,
{
    fn new(
        constitutive_model_parameters: Y,
        reference_nodal_coordinates: ReferenceNodalCoordinates<N>,
        thickness: &Scalar,
    ) -> Self;
}

pub trait FiniteElementMethods<C, const G: usize, const N: usize> {
    fn constitutive_models(&self) -> &[C; G];
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradientList<G>;
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> DeformationGradientRateList<G>;
    fn gradient_vectors(&self) -> &GradientVectors<G, N>;
    fn integration_weights(&self) -> &Scalars<G>;
}

pub trait SurfaceFiniteElementMethods<
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    Self: SurfaceFiniteElementMethodsExtra<M, N, P>,
{
    fn bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P>;
    fn dual_bases<const I: usize>(nodal_coordinates: &Coordinates<I, N>) -> Bases<I, P>;
    fn normals(nodal_coordinates: &NodalCoordinates<N>) -> Normals<P>;
    fn normal_gradients(nodal_coordinates: &NodalCoordinates<N>) -> NormalGradients<N, P>;
    fn normal_rates(
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> NormalRates<P>;
    fn reference_normals(&self) -> &ReferenceNormals<P>;
}

// make this a const fn and remove inherent impl of it once Rust stabilizes const fn trait methods
pub trait SurfaceFiniteElementMethodsExtra<const M: usize, const N: usize, const P: usize> {
    fn standard_gradient_operators() -> StandardGradientOperators<M, N, P>;
}

pub enum FiniteElementError {
    Upstream(String, String),
}

impl Debug for FiniteElementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, element) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In finite element: {element}."
                )
            }
        };
        write!(f, "\n{error}\n\x1b[0;2;31m{}\x1b[0m\n", defeat_message())
    }
}

impl Display for FiniteElementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, element) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In finite element: {element}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

impl<C, const G: usize, const N: usize> FiniteElementMethods<C, G, N> for Element<C, G, N> {
    fn constitutive_models(&self) -> &[C; G] {
        &self.constitutive_models
    }
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> DeformationGradientList<G> {
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
    ) -> DeformationGradientRateList<G> {
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

impl<C, const G: usize, const M: usize, const N: usize, const P: usize>
    SurfaceFiniteElementMethods<G, M, N, P> for SurfaceElement<C, G, N, P>
where
    Self: SurfaceFiniteElementMethodsExtra<M, N, P>,
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

pub trait ElasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Elastic,
    Self: Debug + FiniteElementMethods<C, G, N>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, ConstitutiveError>;
}

pub trait HyperelasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Hyperelastic,
    Self: ElasticFiniteElement<C, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError>;
}

pub trait ViscoelasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Viscoelastic,
    Self: FiniteElementMethods<C, G, N>,
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

pub trait ElasticHyperviscousFiniteElement<C, const G: usize, const N: usize>
where
    C: ElasticHyperviscous,
    Self: ViscoelasticFiniteElement<C, G, N>,
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

pub trait HyperviscoelasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Hyperviscoelastic,
    Self: ElasticHyperviscousFiniteElement<C, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, ConstitutiveError>;
}

impl<C, const G: usize, const N: usize> ElasticFiniteElement<C, G, N> for Element<C, G, N>
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
            Ok(foo) => Ok(foo
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

impl<C, const G: usize, const N: usize> HyperelasticFiniteElement<C, G, N> for Element<C, G, N>
where
    C: Hyperelastic,
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

impl<C, const G: usize, const N: usize> ViscoelasticFiniteElement<C, G, N> for Element<C, G, N>
where
    C: Viscoelastic,
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

impl<C, const G: usize, const N: usize> ElasticHyperviscousFiniteElement<C, G, N>
    for Element<C, G, N>
where
    C: ElasticHyperviscous,
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

impl<C, const G: usize, const N: usize> HyperviscoelasticFiniteElement<C, G, N> for Element<C, G, N>
where
    C: Hyperviscoelastic,
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
