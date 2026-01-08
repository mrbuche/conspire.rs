pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperviscoelastic;
pub mod viscoelastic;
pub mod viscoplastic;

use crate::{
    fem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, FiniteElement,
        surface::{SurfaceElement, SurfaceFiniteElement},
    },
    math::Tensor,
    mechanics::{
        DeformationGradient, DeformationGradientList, DeformationGradientRate,
        DeformationGradientRateList, ForceList, StiffnessList,
    },
};

pub type ElementNodalForcesSolid<const N: usize> = ForceList<N>;
pub type ElementNodalStiffnessesSolid<const N: usize> = StiffnessList<N>;

pub trait SolidFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: FiniteElement<G, M, N, P>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> DeformationGradientList<G>;
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> DeformationGradientRateList<G>;
}

impl<const G: usize, const N: usize, const O: usize, const P: usize> SolidFiniteElement<G, 3, N, P>
    for Element<G, N, O>
where
    Self: FiniteElement<G, 3, N, P>,
{
    fn deformation_gradients(
        &self,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> DeformationGradientList<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_coordinates
                    .iter()
                    .zip(gradient_vectors)
                    .map(|(nodal_coordinate, gradient_vector)| {
                        (nodal_coordinate, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
    fn deformation_gradient_rates(
        &self,
        _: &ElementNodalCoordinates<N>,
        nodal_velocities: &ElementNodalVelocities<N>,
    ) -> DeformationGradientRateList<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_velocities
                    .iter()
                    .zip(gradient_vectors)
                    .map(|(nodal_velocity, gradient_vector)| {
                        (nodal_velocity, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
}

impl<const G: usize, const N: usize, const O: usize> SolidFiniteElement<G, 2, N, N>
    for SurfaceElement<G, N, O>
where
    Self: SurfaceFiniteElement<G, N, N>,
{
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
                    .zip(gradient_vectors)
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
                    .zip(gradient_vectors)
                    .map(|(nodal_velocity, gradient_vector)| {
                        (nodal_velocity, gradient_vector).into()
                    })
                    .sum::<DeformationGradientRate>()
                    + DeformationGradientRate::from(normal_rate_and_reference_normal)
            })
            .collect()
    }
}
