pub mod elastic;
pub mod hyperelastic;

use crate::{
    math::Tensor,
    mechanics::{
        DeformationGradient, DeformationGradientRate, DeformationGradientRates,
        DeformationGradients, Forces, Stiffnesses,
    },
    vem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, VirtualElement,
    },
};

pub use crate::domain::block::element::solid::SolidElement;

pub type ElementNodalForcesSolid = Forces;
pub type ElementNodalStiffnessesSolid = Stiffnesses;

pub trait SolidVirtualElement
where
    Self: VirtualElement
        + SolidElement<
            Coordinates = ElementNodalCoordinates,
            Velocities = ElementNodalVelocities,
            DeformationGradients = DeformationGradients,
            DeformationGradientRates = DeformationGradientRates,
        >,
{
}

impl<T> SolidVirtualElement for T where
    T: VirtualElement
        + SolidElement<
            Coordinates = ElementNodalCoordinates,
            Velocities = ElementNodalVelocities,
            DeformationGradients = DeformationGradients,
            DeformationGradientRates = DeformationGradientRates,
        >
{
}

impl SolidElement for Element
where
    Self: VirtualElement,
{
    type Coordinates = ElementNodalCoordinates;
    type Velocities = ElementNodalVelocities;
    type DeformationGradients = DeformationGradients;
    type DeformationGradientRates = DeformationGradientRates;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &ElementNodalCoordinates,
    ) -> DeformationGradients {
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
        _: &ElementNodalCoordinates,
        nodal_velocities: &ElementNodalVelocities,
    ) -> DeformationGradientRates {
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
