pub mod elastic;
pub mod hyperelastic;

use crate::{
    fem::block::element::{FiniteElement, solid::SolidFiniteElement},
    math::{Scalar, Tensor},
    mechanics::{DeformationGradient, DeformationGradients, Forces, Stiffnesses},
    vem::block::element::{Element, ElementNodalCoordinates, VirtualElement},
};

pub type ElementNodalForcesSolid = Forces;
pub type ElementNodalStiffnessesSolid = Stiffnesses;

pub trait SolidVirtualElement {
    fn deformation_gradients<'a>(
        &'a self,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> DeformationGradients;
    fn tetrahedra_deformation_gradients_and_volumes<'a>(
        &'a self,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Vec<(DeformationGradient, Scalar)>;
}

impl SolidVirtualElement for Element {
    fn deformation_gradients<'a>(
        &'a self,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> DeformationGradients {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                nodal_coordinates
                    .iter()
                    .zip(gradient_vectors)
                    .map(|(&nodal_coordinate, gradient_vector)| {
                        (nodal_coordinate, gradient_vector).into()
                    })
                    .sum()
            })
            .collect()
    }
    fn tetrahedra_deformation_gradients_and_volumes<'a>(
        &'a self,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Vec<(DeformationGradient, Scalar)> {
        self.tetrahedra()
            .iter()
            .map(|tetrahedron| {
                let element_volume: Scalar = tetrahedron.integration_weights().into_iter().sum();
                (
                    tetrahedron
                        .deformation_gradients(todo!())
                        .into_iter()
                        .zip(tetrahedron.integration_weights())
                        .map(|(deformation_gradient, integration_weight)| {
                            deformation_gradient * integration_weight / element_volume
                        })
                        .sum(),
                    element_volume,
                )
            })
            .collect()
    }
}
