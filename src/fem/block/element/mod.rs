#[cfg(test)]
mod test;

pub mod composite;
pub mod linear;
pub mod solid;
pub mod thermal;

pub use self::solid::{
    SolidFiniteElement, elastic::ElasticFiniteElement,
    elastic_hyperviscous::ElasticHyperviscousFiniteElement,
    elastic_viscoplastic::ElasticViscoplasticFiniteElement,
    hyperelastic::HyperelasticFiniteElement, hyperviscoelastic::HyperviscoelasticFiniteElement,
    viscoelastic::ViscoelasticFiniteElement, viscoplastic::ViscoplasticStateVariables,
};

use crate::{
    defeat_message,
    fem::{
        Bases, GradientVectors, NormalGradients, NormalRates, Normals, ReferenceNormals,
        StandardGradientOperators,
    },
    math::{IDENTITY, LEVI_CIVITA, Scalar, Scalars, Tensor, TensorArray, TensorRank2, TestError},
    mechanics::{Coordinates, CurrentCoordinates, Normal, ReferenceCoordinates},
};
use std::fmt::{self, Debug, Display, Formatter};

pub type ElementNodalCoordinates<const N: usize> = CurrentCoordinates<N>;
pub type ElementNodalVelocities<const N: usize> = CurrentCoordinates<N>;
pub type ElementReferenceNodalCoordinates<const N: usize> = ReferenceCoordinates<N>;

pub struct Element<const G: usize, const N: usize> {
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars<G>,
}

impl<const G: usize, const N: usize> Element<G, N> {
    fn gradient_vectors(&self) -> &GradientVectors<G, N> {
        &self.gradient_vectors
    }
    fn integration_weights(&self) -> &Scalars<G> {
        &self.integration_weights
    }
}

impl<const G: usize, const N: usize> From<ElementReferenceNodalCoordinates<N>> for Element<G, N>
where
    Self: FiniteElement<G, N>,
{
    fn from(reference_nodal_coordinates: ElementReferenceNodalCoordinates<N>) -> Self {
        let (gradient_vectors, integration_weights) = Self::initialize(reference_nodal_coordinates);
        Self {
            gradient_vectors,
            integration_weights,
        }
    }
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

pub trait FiniteElement<const G: usize, const N: usize>
where
    Self: From<ElementReferenceNodalCoordinates<N>>,
{
    fn initialize(
        reference_nodal_coordinates: ElementReferenceNodalCoordinates<N>,
    ) -> (GradientVectors<G, N>, Scalars<G>);
    fn reset(&mut self);
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

pub struct SurfaceElement<const G: usize, const N: usize, const P: usize> {
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars<G>,
    reference_normals: ReferenceNormals<P>,
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

impl<const G: usize, const N: usize, const P: usize> SurfaceElement<G, N, P> {
    fn integration_weights(&self) -> &Scalars<G> {
        &self.integration_weights
    }
    fn reference_normals(&self) -> &ReferenceNormals<P> {
        &self.reference_normals
    }
}

pub trait SurfaceFiniteElement<const G: usize, const N: usize, const P: usize> {
    fn new(
        reference_nodal_coordinates: ElementReferenceNodalCoordinates<N>,
        thickness: Scalar,
    ) -> Self;
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
    fn normals(nodal_coordinates: &ElementNodalCoordinates<N>) -> Normals<P>;
    fn normal_gradients(nodal_coordinates: &ElementNodalCoordinates<N>) -> NormalGradients<N, P>;
    fn normal_rates(
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> NormalRates<P>;
}

// make this a const fn and remove inherent impl of it once Rust stabilizes const fn trait methods
pub trait SurfaceFiniteElementMethodsExtra<const M: usize, const N: usize, const P: usize> {
    fn standard_gradient_operators() -> StandardGradientOperators<M, N, P>;
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
    fn normals(nodal_coordinates: &ElementNodalCoordinates<N>) -> Normals<P> {
        Self::bases(nodal_coordinates)
            .iter()
            .map(|basis_vectors| basis_vectors[0].cross(&basis_vectors[1]).normalized())
            .collect()
    }
    fn normal_gradients(nodal_coordinates: &ElementNodalCoordinates<N>) -> NormalGradients<N, P> {
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization: Scalar = 0.0;
        let mut normal_vector = Normal::zero();
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
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
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
