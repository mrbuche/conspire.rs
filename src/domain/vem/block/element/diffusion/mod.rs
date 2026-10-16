pub mod conduction;

pub use conduction::{DiffusionVirtualElement, ElementNodalForces, ElementNodalStiffnesses};

use crate::{
    constitutive::diffusion::Diffusive,
    domain::diffusion::NodalFields,
    fem::block::element::diffusion::ElementNodalFields as TetrahedronNodalFields,
    math::{Quantity, Reference, Scalar, Tensor, TensorRank1},
    units::{ReciprocalLength, UnitMul},
    vem::block::element::{Element, VirtualElement},
};

pub trait DiffusiveVirtualElement<L>
where
    L: Diffusive,
    Self: VirtualElement,
{
    fn gradient(&self, nodal_fields: &NodalFields<L>) -> TensorRank1<3, Reference, L::Gradient>;
    fn tetrahedra_fields(&self, nodal_fields: &NodalFields<L>)
    -> Vec<TetrahedronNodalFields<L, 4>>;
}

fn mean<U>(fields: Vec<Quantity<U>>) -> Quantity<U> {
    let number = fields.len() as Scalar;
    fields.into_iter().sum::<Quantity<U>>() / number
}

impl<L> DiffusiveVirtualElement<L> for Element
where
    L: Diffusive,
    ReciprocalLength: UnitMul<L::Field, Output = L::Gradient>,
{
    fn gradient(&self, nodal_fields: &NodalFields<L>) -> TensorRank1<3, Reference, L::Gradient> {
        nodal_fields
            .iter()
            .zip(&self.gradient_vectors()[0])
            .map(|(nodal_field, gradient_vector)| gradient_vector * nodal_field)
            .sum()
    }
    fn tetrahedra_fields(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Vec<TetrahedronNodalFields<L, 4>> {
        let element_center = mean(nodal_fields.iter().copied().collect());
        let faces_centers = self
            .faces_nodes()
            .iter()
            .map(|face_nodes| mean(face_nodes.iter().map(|&node| nodal_fields[node]).collect()))
            .collect::<Vec<_>>();
        self.tetrahedra_nodes()
            .iter()
            .map(|&[face, node_b, node_a]| {
                [
                    faces_centers[face],
                    nodal_fields[node_b],
                    nodal_fields[node_a],
                    element_center,
                ]
                .into()
            })
            .collect()
    }
}
