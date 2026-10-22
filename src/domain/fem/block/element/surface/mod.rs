pub mod linear;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalEitherCoordinates, ElementNodalReferenceCoordinates,
        ElementNodalVelocities, FiniteElement, FiniteElementIntegration, GradientVectors,
        IntegrationWeights,
    },
    math::{
        CrossProduct, Current, IDENTITY, LEVI_CIVITA, Quantity, Tensor, TensorArray, TensorRank1,
        TensorRank2, TensorRank2List2D,
    },
    mechanics::{
        ForceList, Normal, NormalGradients, NormalRates, Normals, ReferenceNormals, SurfaceBases,
        SurfaceDualBases, Traction,
    },
    units::{Area, ForcePerLength, Length, Rate, Stress, Volume},
};
use std::fmt::{self, Debug, Formatter};

const M: usize = 2;

#[derive(Clone)]
pub struct SurfaceElement<const G: usize, const N: usize, const O: usize> {
    gradient_vectors: GradientVectors<3, G, N>,
    integration_weights: IntegrationWeights<G, Volume>,
    reference_normals: ReferenceNormals<G>,
}

impl<const G: usize, const N: usize, const O: usize> SurfaceElement<G, N, O> {
    pub fn gradient_vectors(&self) -> &GradientVectors<3, G, N> {
        &self.gradient_vectors
    }
    pub fn reference_normals(&self) -> &ReferenceNormals<G> {
        &self.reference_normals
    }
}

impl<const G: usize, const N: usize, const O: usize> FiniteElementIntegration<G, Volume>
    for SurfaceElement<G, N, O>
{
    fn integration_weights(&self) -> &IntegrationWeights<G, Volume> {
        &self.integration_weights
    }
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

pub trait SurfaceFiniteElement<const G: usize, const N: usize, const P: usize, W = Volume>
where
    Self: FiniteElement<G, M, N, P> + FiniteElementIntegration<G, W>,
{
    fn bases<I>(nodal_coordinates: &ElementNodalEitherCoordinates<I, P>) -> SurfaceBases<I, G> {
        Self::shape_functions_gradients_at_integration_points()
            .iter()
            .map(|shape_functions_gradients| {
                shape_functions_gradients
                    .iter()
                    .zip(nodal_coordinates)
                    .map(|(shape_functions_gradient, nodal_coordinate)| {
                        shape_functions_gradient
                            .iter()
                            .map(|standard_gradient_operator_m| {
                                nodal_coordinate * standard_gradient_operator_m
                            })
                            .collect()
                    })
                    .sum()
            })
            .collect()
    }
    fn dual_bases<I>(
        nodal_coordinates: &ElementNodalEitherCoordinates<I, P>,
    ) -> SurfaceDualBases<I, G> {
        Self::bases(nodal_coordinates)
            .into_iter()
            .map(|basis_vectors| {
                basis_vectors
                    .iter()
                    .map(|basis_vector_m| {
                        basis_vectors
                            .iter()
                            .map(|basis_vector_n| basis_vector_m * basis_vector_n)
                            .collect()
                    })
                    .collect::<TensorRank2<2, I, I, Area>>()
                    .inverse()
                    .iter()
                    .map(|metric_tensor_m| {
                        metric_tensor_m
                            .iter()
                            .zip(basis_vectors.iter())
                            .map(|(metric_tensor_mn, basis_vectors_n)| {
                                basis_vectors_n * *metric_tensor_mn
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
    fn nodal_areas(
        reference_coordinates: &ElementNodalReferenceCoordinates<P>,
    ) -> Vec<Quantity<Area>> {
        let shape_functions = Self::shape_functions_at_integration_points();
        let areas = Self::bases(reference_coordinates)
            .into_iter()
            .zip(Self::parametric_weights())
            .map(|(basis, parametric_weight)| basis[0].cross(&basis[1]).norm() * parametric_weight)
            .collect::<Vec<_>>();
        (0..P)
            .map(|a| {
                shape_functions
                    .iter()
                    .zip(&areas)
                    .map(|(shape_functions, &area)| area * shape_functions[a])
                    .sum()
            })
            .collect()
    }
    fn traction_forces(
        reference_coordinates: &ElementNodalReferenceCoordinates<P>,
        traction: &Traction,
    ) -> ForceList<P> {
        Self::nodal_areas(reference_coordinates)
            .into_iter()
            .map(|area| traction * area)
            .collect()
    }
    /// The forces of a pressure on the face, which pushes into the body and follows the face as
    /// it moves, so that they depend on the current coordinates.
    ///
    /// The nodes of the face are numbered counterclockwise as seen from outside the body.
    fn pressure_forces(
        nodal_coordinates: &ElementNodalCoordinates<P>,
        pressure: Quantity<Stress>,
    ) -> ForceList<P> {
        let shape_functions = Self::shape_functions_at_integration_points();
        let normals = Self::bases(nodal_coordinates)
            .into_iter()
            .zip(Self::parametric_weights())
            .map(|(basis, parametric_weight)| basis[0].cross(&basis[1]) * parametric_weight)
            .collect::<Vec<_>>();
        (0..P)
            .map(|a| {
                shape_functions
                    .iter()
                    .zip(&normals)
                    .map(|(shape_functions, normal)| normal * shape_functions[a])
                    .sum::<TensorRank1<3, Current, Area>>()
                    * -pressure
            })
            .collect()
    }
    /// The derivatives of the forces of a pressure on the face with respect to its coordinates.
    fn pressure_stiffnesses(
        nodal_coordinates: &ElementNodalCoordinates<P>,
        pressure: Quantity<Stress>,
    ) -> TensorRank2List2D<3, Current, Current, P, P, ForcePerLength> {
        let shape_functions = Self::shape_functions_at_integration_points();
        let gradients = Self::shape_functions_gradients_at_integration_points();
        let bases = Self::bases(nodal_coordinates);
        let weights = Self::parametric_weights();
        let zero = Quantity::<Length>::new(0.0);
        let skew = |vector: &TensorRank1<3, Current, Length>| {
            [
                [zero, -vector[2], vector[1]],
                [vector[2], zero, -vector[0]],
                [-vector[1], vector[0], zero],
            ]
        };
        (0..P)
            .map(|a| {
                (0..P)
                    .map(|b| {
                        let mut block = [[Quantity::<ForcePerLength>::new(0.0); 3]; 3];
                        shape_functions
                            .iter()
                            .zip(gradients.iter())
                            .zip(bases.iter())
                            .zip(weights.iter())
                            .for_each(|(((shape_functions, gradients), basis), &weight)| {
                                let (skew_1, skew_2) = (skew(&basis[0]), skew(&basis[1]));
                                let scale = -weight * shape_functions[a].value();
                                let (gradient_1, gradient_2) =
                                    (gradients[b][0].value(), gradients[b][1].value());
                                (0..3).for_each(|i| {
                                    (0..3).for_each(|j| {
                                        block[i][j] += pressure
                                            * (skew_1[i][j] * gradient_2
                                                - skew_2[i][j] * gradient_1)
                                            * scale
                                    })
                                })
                            });
                        TensorRank2::from(block)
                    })
                    .collect()
            })
            .collect()
    }
    fn normals(nodal_coordinates: &ElementNodalCoordinates<P>) -> Normals<G> {
        Self::bases(nodal_coordinates)
            .into_iter()
            .map(|basis| {
                let normal = basis[0].cross(&basis[1]);
                let area = normal.norm();
                normal / area
            })
            .collect()
    }
    fn normal_gradients(nodal_coordinates: &ElementNodalCoordinates<P>) -> NormalGradients<P, G> {
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization = Quantity::<Area>::new(0.0);
        let mut normal_vector = Normal::zero();
        Self::shape_functions_gradients_at_integration_points().iter()
        .zip(Self::bases(nodal_coordinates))
        .map(|(standard_gradient_operator, basis_vectors)|{
            normalization = basis_vectors[0].cross(&basis_vectors[1]).norm();
            normal_vector = basis_vectors[0].cross(&basis_vectors[1]) / normalization;
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
                            ).sum::<Quantity>() * (
                                standard_gradient_operator_a[0] * basis_vector_1_n
                              - standard_gradient_operator_a[1] * basis_vector_0_n
                            )
                        ).sum::<Quantity<Length>>()
                            / normalization
                    ).collect()
                ).collect()
            ).collect()
        }).collect()
    }
    fn normal_rates(
        nodal_coordinates: &ElementNodalCoordinates<P>,
        nodal_velocities: &ElementNodalVelocities<P>,
    ) -> NormalRates<G> {
        let identity = IDENTITY;
        let levi_civita_symbol = LEVI_CIVITA;
        let mut normalization = Quantity::<Area>::new(0.0);
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
                                ).sum::<Quantity>() * (
                                    standard_gradient_operator_a[0] * basis_vector_1_n
                                - standard_gradient_operator_a[1] * basis_vector_0_n
                                )
                            ).sum::<Quantity<Length>>()
                            / normalization
                                * nodal_velocity_a_m
                        ).sum::<Quantity<Rate>>()
                    ).sum::<Quantity<Rate>>()
                ).collect()
        }).collect()
    }
}

impl<const G: usize, const N: usize, const O: usize, const P: usize> SurfaceFiniteElement<G, N, P>
    for SurfaceElement<G, N, O>
where
    Self: FiniteElement<G, M, N, P>,
{
}

impl<const G: usize, const N: usize, const O: usize>
    From<(ElementNodalReferenceCoordinates<N>, Quantity<Length>)> for SurfaceElement<G, N, O>
where
    Self: SurfaceFiniteElement<G, N, N>,
{
    fn from(
        (reference_nodal_coordinates, thickness): (
            ElementNodalReferenceCoordinates<N>,
            Quantity<Length>,
        ),
    ) -> Self {
        let integration_weights = Self::bases(&reference_nodal_coordinates)
            .into_iter()
            .zip(Self::parametric_weights())
            .map(|(reference_basis, parametric_weight)| {
                reference_basis[0].cross(&reference_basis[1]).norm() * parametric_weight * thickness
            })
            .collect();
        let reference_dual_bases = Self::dual_bases(&reference_nodal_coordinates);
        let gradient_vectors = Self::shape_functions_gradients_at_integration_points()
            .into_iter()
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
            .into_iter()
            .map(|reference_dual_basis| {
                let normal = reference_dual_basis[0].cross(&reference_dual_basis[1]);
                let reciprocal_area = normal.norm();
                normal / reciprocal_area
            })
            .collect();
        Self {
            gradient_vectors,
            integration_weights,
            reference_normals,
        }
    }
}
