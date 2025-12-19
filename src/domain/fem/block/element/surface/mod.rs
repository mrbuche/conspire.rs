pub mod linear;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
        FiniteElement, GradientVectors,
    },
    math::{IDENTITY, LEVI_CIVITA, Scalar, ScalarList, Tensor, TensorArray, TensorRank2},
    mechanics::{
        CoordinateList, Normal, NormalGradients, NormalRates, Normals, ReferenceNormals,
        SurfaceBases,
    },
};
use std::fmt::{self, Debug, Formatter};

pub struct SurfaceElement<const G: usize, const N: usize> {
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: ScalarList<G>,
    reference_normals: ReferenceNormals<G>,
}

impl<const G: usize, const N: usize> Debug for SurfaceElement<G, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N) {
            (1, 3) => "LinearTriangle",
            _ => panic!(),
        };
        write!(f, "{element} {{ G: {G}, N: {N} }}",)
    }
}

impl<const G: usize, const N: usize> SurfaceElement<G, N> {
    fn gradient_vectors(&self) -> &GradientVectors<G, N> {
        &self.gradient_vectors
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn reference_normals(&self) -> &ReferenceNormals<G> {
        &self.reference_normals
    }
}

impl<const G: usize, const N: usize> Default for SurfaceElement<G, N>
where
    Self: FiniteElement<G, 2, N> + From<(ElementNodalReferenceCoordinates<N>, Scalar)>,
{
    fn default() -> Self {
        (Self::parametric_reference(), 1.0).into()
    }
}

pub trait SurfaceFiniteElementCreation<const G: usize, const N: usize>
where
    Self: Default + From<(ElementNodalReferenceCoordinates<N>, Scalar)>,
{
}

impl<const G: usize, const N: usize> SurfaceFiniteElementCreation<G, N> for SurfaceElement<G, N> where
    Self: Default + From<(ElementNodalReferenceCoordinates<N>, Scalar)>
{
}

pub trait SurfaceFiniteElementMethods<const G: usize, const M: usize, const N: usize>
where
    Self: FiniteElement<G, M, N>,
{
    fn bases<const I: usize>(nodal_coordinates: &CoordinateList<I, N>) -> SurfaceBases<I, G>;
    fn dual_bases<const I: usize>(nodal_coordinates: &CoordinateList<I, N>) -> SurfaceBases<I, G>;
    fn normals(nodal_coordinates: &ElementNodalCoordinates<N>) -> Normals<G>;
    fn normal_gradients(nodal_coordinates: &ElementNodalCoordinates<N>) -> NormalGradients<N, G>;
    fn normal_rates(
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> NormalRates<G>;
}

impl<const G: usize, const M: usize, const N: usize> SurfaceFiniteElementMethods<G, M, N>
    for SurfaceElement<G, N>
where
    Self: FiniteElement<G, M, N>,
{
    fn bases<const I: usize>(nodal_coordinates: &CoordinateList<I, N>) -> SurfaceBases<I, G> {
        Self::shape_functions_gradients_at_integration_points()
            .iter()
            .map(|standard_gradient_operator| {
                standard_gradient_operator
                    .iter()
                    .zip(nodal_coordinates)
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
    fn dual_bases<const I: usize>(nodal_coordinates: &CoordinateList<I, N>) -> SurfaceBases<I, G> {
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
                            .zip(basis_vectors)
                            .map(|(metric_tensor_mn, basis_vectors_n)| {
                                basis_vectors_n * metric_tensor_mn
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
    fn normals(nodal_coordinates: &ElementNodalCoordinates<N>) -> Normals<G> {
        Self::bases(nodal_coordinates)
            .iter()
            .map(|basis_vectors| basis_vectors[0].cross(&basis_vectors[1]).normalized())
            .collect()
    }
    fn normal_gradients(nodal_coordinates: &ElementNodalCoordinates<N>) -> NormalGradients<N, G> {
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization: Scalar = 0.0;
        let mut normal_vector = Normal::zero();
        Self::shape_functions_gradients_at_integration_points().iter()
        .zip(Self::bases(nodal_coordinates))
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
    ) -> NormalRates<G> {
        let identity = IDENTITY;
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization = 0.0;
        Self::bases(nodal_coordinates)
            .iter()
            .zip(Self::normals(nodal_coordinates).iter()
            .zip(Self::shape_functions_gradients_at_integration_points()))
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
