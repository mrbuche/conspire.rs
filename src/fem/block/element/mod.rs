#[cfg(test)]
mod test;

pub mod composite;
pub mod linear;

use super::*;
use crate::{
    defeat_message,
    math::{IDENTITY, LEVI_CIVITA, TensorTupleList, tensor_rank_1_zero},
    mechanics::{HeatFluxes, Scalar, TemperatureGradients},
};
use std::fmt::{Debug, Display};

// pub struct Foo<const G: usize, T> {
//     bar: T,
//     integration_weights: Scalars<G>,
// }

pub struct Element<const G: usize, const N: usize> {
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars<G>,
}

pub struct SurfaceElement<const G: usize, const N: usize, const P: usize> {
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars<G>,
    reference_normals: ReferenceNormals<P>,
}

impl<const G: usize, const N: usize> Debug for Element<G, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N) {
            (1, 4) => "LinearTetrahedron",
            (8, 8) => "LinearHexahedron",
            (4, 10) => "CompositeTetrahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ G: {G}, N: {N} }}",)
    }
}

impl<const G: usize, const N: usize, const P: usize> Debug for SurfaceElement<G, N, P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N, P) {
            (1, 3, 1) => "LinearTriangle",
            _ => panic!(),
        };
        write!(f, "{element} {{ G: {G}, N: {N} , P: {P} }}",)
    }
}

impl<const G: usize, const N: usize> Element<G, N> {
    fn integration_weights(&self) -> &Scalars<G> {
        &self.integration_weights
    }
}

impl<const G: usize, const N: usize, const P: usize> SurfaceElement<G, N, P> {
    fn integration_weights(&self) -> &Scalars<G> {
        &self.integration_weights
    }
    fn reference_normals(&self) -> &ReferenceNormals<P> {
        &self.reference_normals
    }
}

pub trait FiniteElement<const G: usize, const N: usize>
{
    fn new(reference_nodal_coordinates: ReferenceNodalCoordinates<N>) -> Self;
    fn reference() -> ReferenceNodalCoordinates<N>;
    fn reset(&mut self);
}

pub trait SurfaceFiniteElement<const G: usize, const N: usize, const P: usize>
{
    fn new(reference_nodal_coordinates: ReferenceNodalCoordinates<N>, thickness: Scalar) -> Self;
}

pub trait FiniteElementMethods<const G: usize, const N: usize> {
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
}

// make this a const fn and remove inherent impl of it once Rust stabilizes const fn trait methods
pub trait SurfaceFiniteElementMethodsExtra<const M: usize, const N: usize, const P: usize> {
    fn standard_gradient_operators() -> StandardGradientOperators<M, N, P>;
}

pub enum FiniteElementError {
    Upstream(String, String),
}

impl From<FiniteElementError> for TestError {
    fn from(error: FiniteElementError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
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

impl<const G: usize, const N: usize> FiniteElementMethods<G, N> for Element<G, N> {
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
                        (nodal_coordinate, gradient_vector).into()
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
                        (nodal_velocity, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
    fn gradient_vectors(&self) -> &GradientVectors<G, N> {
        &self.gradient_vectors
    }
}

impl<const G: usize, const M: usize, const N: usize, const P: usize>
    SurfaceFiniteElementMethods<G, M, N, P> for SurfaceElement<G, N, P>
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
}

pub trait ElasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Elastic,
    Self: Debug + FiniteElementMethods<G, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError>;
}

pub trait HyperelasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Hyperelastic,
    Self: ElasticFiniteElement<C, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError>;
}

pub type ViscoplasticStateVariables<const G: usize> =
    TensorTupleList<DeformationGradientPlastic, Scalar, G>;

pub trait ElasticViscoplasticFiniteElement<C, const G: usize, const N: usize>
where
    C: ElasticViscoplastic,
    Self: Debug + FiniteElementMethods<G, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalForces<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError>;
    fn state_variables_evolution(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ViscoplasticStateVariables<G>, FiniteElementError>;
}

pub trait ViscoelasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Viscoelastic,
    Self: FiniteElementMethods<G, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError>;
}

pub trait ElasticHyperviscousFiniteElement<C, const G: usize, const N: usize>
where
    C: ElasticHyperviscous,
    Self: ViscoelasticFiniteElement<C, G, N>,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError>;
    fn dissipation_potential(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError>;
}

pub trait HyperviscoelasticFiniteElement<C, const G: usize, const N: usize>
where
    C: Hyperviscoelastic,
    Self: ElasticHyperviscousFiniteElement<C, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError>;
}

impl<C, const G: usize, const N: usize> ElasticFiniteElement<C, G, N> for Element<G, N>
where
    C: Elastic,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalForces<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .map(|deformation_gradient| {
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
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses<G>, _>>()
        {
            Ok(first_piola_kirchhoff_tangent_stiffnesses) => {
                Ok(first_piola_kirchhoff_tangent_stiffnesses
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
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, const G: usize, const N: usize> HyperelasticFiniteElement<C, G, N> for Element<G, N>
where
    C: Hyperelastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(self.integration_weights().iter())
            .map(|(deformation_gradient, integration_weight)| {
                Ok::<Scalar, ConstitutiveError>(
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

impl<C, const G: usize, const N: usize> ElasticViscoplasticFiniteElement<C, G, N> for Element<G, N>
where
    C: ElasticViscoplastic,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalForces<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables.iter())
            .map(|(deformation_gradient, state_variable)| {
                let (deformation_gradient_p, _) = state_variable.into();
                constitutive_model
                    .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)
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
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables.iter())
            .map(|(deformation_gradient, state_variable)| {
                let (deformation_gradient_p, _) = state_variable.into();
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(
                    deformation_gradient,
                    deformation_gradient_p,
                )
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses<G>, _>>()
        {
            Ok(first_piola_kirchhoff_tangent_stiffnesses) => {
                Ok(first_piola_kirchhoff_tangent_stiffnesses
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
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn state_variables_evolution(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<ViscoplasticStateVariables<G>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables.iter())
            .map(|(deformation_gradient, state_variable)| {
                constitutive_model.state_variables_evolution(deformation_gradient, state_variable)
            })
            .collect::<Result<ViscoplasticStateVariables<G>, _>>()
        {
            Ok(state_variables_evolution) => Ok(state_variables_evolution),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, const G: usize, const N: usize> ViscoelasticFiniteElement<C, G, N> for Element<G, N>
where
    C: Viscoelastic,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalForces<N>, FiniteElementError> {
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
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<NodalStiffnesses<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter(),
            )
            .map(|(deformation_gradient, deformation_gradient_rate)| {
                constitutive_model.first_piola_kirchhoff_rate_tangent_stiffness(
                    deformation_gradient,
                    deformation_gradient_rate,
                )
            })
            .collect::<Result<FirstPiolaKirchhoffRateTangentStiffnesses<G>, _>>()
        {
            Ok(first_piola_kirchhoff_rate_tangent_stiffnesses) => {
                Ok(first_piola_kirchhoff_rate_tangent_stiffnesses
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
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<C, const G: usize, const N: usize> ElasticHyperviscousFiniteElement<C, G, N> for Element<G, N>
where
    C: ElasticHyperviscous,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights().iter()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
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
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
        nodal_velocities: &NodalVelocities<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights().iter()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
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

impl<C, const G: usize, const N: usize> HyperviscoelasticFiniteElement<C, G, N> for Element<G, N>
where
    C: Hyperviscoelastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<N>,
    ) -> Result<Scalar, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(self.integration_weights().iter())
            .map(|(deformation_gradient, integration_weight)| {
                Ok::<Scalar, ConstitutiveError>(
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

pub trait ThermalConductionFiniteElement<C, const G: usize, const N: usize>
where
    C: ThermalConduction,
    Self: Debug + FiniteElementMethods<G, N>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> Result<NodalForcesThermal<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> Result<NodalStiffnessesThermal<N>, FiniteElementError>;
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> TemperatureGradients<G>;
}

impl<C, const G: usize, const N: usize> ThermalConductionFiniteElement<C, G, N> for Element<G, N>
where
    C: ThermalConduction,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> Result<NodalForcesThermal<N>, FiniteElementError> {
        todo!()
        // match self
        //     .temperature_gradients(nodal_temperatures)
        //     .iter()
        //     .map(|temperature_gradient| {
        //         constitutive_model.heat_flux(temperature_gradient)
        //     })
        //     .collect::<Result<HeatFluxes<G>, _>>()
        // {
        //     Ok(heat_fluxes) => todo!(),
        //     // Ok(heat_fluxes) => Ok(heat_fluxes
        //     //     .iter()
        //     //     .zip(
        //     //         self.gradient_vectors()
        //     //             .iter()
        //     //             .zip(self.integration_weights().iter()),
        //     //     )
        //     //     .map(
        //     //         |(heat_flux, (gradient_vectors, integration_weight))| {
        //     //             gradient_vectors
        //     //                 .iter()
        //     //                 .map(|gradient_vector| {
        //     //                     (heat_flux * gradient_vector)
        //     //                         * integration_weight
        //     //                 })
        //     //                 .collect()
        //     //         },
        //     //     )
        //     //     .sum()),
        //     Err(error) => Err(FiniteElementError::Upstream(
        //         format!("{error}"),
        //         format!("{self:?}"),
        //     )),
        // }
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> Result<NodalStiffnessesThermal<N>, FiniteElementError> {
        todo!()
        // match self
        //     .deformation_gradients(nodal_coordinates)
        //     .iter()
        //     .map(|deformation_gradient| {
        //         constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
        //     })
        //     .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses<G>, _>>()
        // {
        //     Ok(first_piola_kirchhoff_tangent_stiffnesses) => {
        //         Ok(first_piola_kirchhoff_tangent_stiffnesses
        //             .iter()
        //             .zip(
        //                 self.gradient_vectors()
        //                     .iter()
        //                     .zip(self.integration_weights().iter()),
        //             )
        //             .map(
        //                 |(
        //                     first_piola_kirchhoff_tangent_stiffness,
        //                     (gradient_vectors, integration_weight),
        //                 )| {
        //                     gradient_vectors
        //                         .iter()
        //                         .map(|gradient_vector_a| {
        //                             gradient_vectors
        //                                 .iter()
        //                                 .map(|gradient_vector_b| {
        //                                     first_piola_kirchhoff_tangent_stiffness
        //                                     .contract_second_fourth_indices_with_first_indices_of(
        //                                         gradient_vector_a,
        //                                         gradient_vector_b,
        //                                     )
        //                                     * integration_weight
        //                                 })
        //                                 .collect()
        //                         })
        //                         .collect()
        //                 },
        //             )
        //             .sum())
        //     }
        //     Err(error) => Err(FiniteElementError::Upstream(
        //         format!("{error}"),
        //         format!("{self:?}"),
        //     )),
        // }
    }
    fn temperature_gradients(
        &self,
        nodal_temperatures: &NodalTemperatures<N>,
    ) -> TemperatureGradients<G> {
        //
        // deformation_gradients(), deformation_gradient_rates(), etc. should be impl for Solid
        // similarly for element blocks
        // gradient_vectors should be a field only for Solid elements too
        // and then something else a field for thermal elements
        // also new() will have to depend on <C>
        //
        todo!()
    }
}
