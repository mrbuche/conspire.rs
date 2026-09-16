pub mod elastic;
pub mod elastic_hyperviscous;
pub mod elastic_viscoplastic;
pub mod hyperelastic;
pub mod hyperelastic_viscoplastic;
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
        DampingList2D, DeformationGradient, DeformationGradientList, DeformationGradientRate,
        DeformationGradientRateList, ForceList, StiffnessList2D,
    },
};

pub use crate::domain::block::element::solid::SolidElement;

pub type ElementNodalForcesSolid<const N: usize> = ForceList<N>;
pub type ElementNodalStiffnessesSolid<const N: usize> = StiffnessList2D<N>;
pub type ElementNodalDampingsSolid<const N: usize> = DampingList2D<N>;

pub trait SolidFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: FiniteElement<G, M, N, P>
        + SolidElement<
            Coordinates = ElementNodalCoordinates<N>,
            Velocities = ElementNodalVelocities<N>,
            DeformationGradients = DeformationGradientList<G>,
            DeformationGradientRates = DeformationGradientRateList<G>,
        >,
{
}

impl<T, const G: usize, const M: usize, const N: usize, const P: usize>
    SolidFiniteElement<G, M, N, P> for T
where
    T: FiniteElement<G, M, N, P>
        + SolidElement<
            Coordinates = ElementNodalCoordinates<N>,
            Velocities = ElementNodalVelocities<N>,
            DeformationGradients = DeformationGradientList<G>,
            DeformationGradientRates = DeformationGradientRateList<G>,
        >,
{
}

impl<const G: usize, const N: usize, const O: usize> SolidElement for Element<3, G, N, O> {
    type Coordinates = ElementNodalCoordinates<N>;
    type Velocities = ElementNodalVelocities<N>;
    type DeformationGradients = DeformationGradientList<G>;
    type DeformationGradientRates = DeformationGradientRateList<G>;
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
                        DeformationGradient::from((nodal_coordinate, gradient_vector))
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
                        DeformationGradientRate::from((nodal_velocity, gradient_vector))
                    })
                    .sum()
            })
            .collect()
    }
}

impl<const G: usize, const N: usize, const O: usize> SolidElement for SurfaceElement<G, N, O>
where
    Self: SurfaceFiniteElement<G, N, N>,
{
    type Coordinates = ElementNodalCoordinates<N>;
    type Velocities = ElementNodalVelocities<N>;
    type DeformationGradients = DeformationGradientList<G>;
    type DeformationGradientRates = DeformationGradientRateList<G>;
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
                        DeformationGradient::from((nodal_coordinate, gradient_vector))
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
                        DeformationGradientRate::from((nodal_velocity, gradient_vector))
                    })
                    .sum::<DeformationGradientRate>()
                    + DeformationGradientRate::from(normal_rate_and_reference_normal)
            })
            .collect()
    }
}
