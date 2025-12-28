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

const M: usize = 2;

pub struct SurfaceElement<const G: usize, const N: usize, const O: usize> {
    gradient_vectors: GradientVectors<G, N>,
    integration_weights: ScalarList<G>,
    reference_normals: ReferenceNormals<G>,
}

impl<const G: usize, const N: usize, const O: usize> Debug for SurfaceElement<G, N, O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N, O) {
            (1, 3, 1) => "LinearTriangle",
            (4, 4, 1) => "LinearQuadrilateral",
            _ => panic!(),
        };
        write!(f, "{element} {{ G: {G}, N: {N} }}",)
    }
}

impl<const G: usize, const N: usize, const O: usize> Default for SurfaceElement<G, N, O>
where
    Self: FiniteElement<G, M, N> + From<(ElementNodalReferenceCoordinates<N>, Scalar)>,
{
    fn default() -> Self {
        // (Self::parametric_reference(), 1.0).into()
        todo!()
    }
}

pub trait SurfaceFiniteElementCreation<const G: usize, const N: usize>
where
    Self: Default + From<(ElementNodalReferenceCoordinates<N>, Scalar)>,
{
}

impl<const G: usize, const N: usize, const O: usize> SurfaceFiniteElementCreation<G, N>
    for SurfaceElement<G, N, O>
where
    Self: Default + From<(ElementNodalReferenceCoordinates<N>, Scalar)>,
{
}

pub trait SurfaceFiniteElement<const G: usize, const N: usize>
where
    Self: FiniteElement<G, M, N>,
{
    fn bases<const I: usize>(nodal_coordinates: &CoordinateList<I, N>) -> SurfaceBases<I, G>;
    fn dual_bases<const I: usize>(nodal_coordinates: &CoordinateList<I, N>) -> SurfaceBases<I, G>;
    fn gradient_vectors(&self) -> &GradientVectors<G, N>;
    fn normals(nodal_coordinates: &ElementNodalCoordinates<N>) -> Normals<G>;
    fn normal_gradients(nodal_coordinates: &ElementNodalCoordinates<N>) -> NormalGradients<N, G>;
    fn normal_rates(
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> NormalRates<G>;
    fn reference_normals(&self) -> &ReferenceNormals<G>;
}

impl<const G: usize, const N: usize, const O: usize> SurfaceFiniteElement<G, N>
    for SurfaceElement<G, N, O>
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
    fn gradient_vectors(&self) -> &GradientVectors<G, N> {
        &self.gradient_vectors
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
    fn reference_normals(&self) -> &ReferenceNormals<G> {
        &self.reference_normals
    }
}

impl<const G: usize, const N: usize, const O: usize>
    From<(ElementNodalReferenceCoordinates<N>, Scalar)> for SurfaceElement<G, N, O>
where
    Self: SurfaceFiniteElement<G, N>,
{
    fn from(
        (reference_nodal_coordinates, thickness): (ElementNodalReferenceCoordinates<N>, Scalar),
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
