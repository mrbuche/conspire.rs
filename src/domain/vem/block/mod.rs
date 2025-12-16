pub mod element;
pub mod solid;

use crate::{
    math::TensorRank1Vec2D,
    mechanics::Coordinates,
    vem::{NodalReferenceCoordinates, block::element::VirtualElement},
};
use std::{
    any::type_name,
    fmt::{self, Debug, Formatter},
};

pub type Connectivity = Vec<Vec<usize>>;

pub struct Block<C, F> {
    constitutive_model: C,
    coordinates: NodalReferenceCoordinates,
    elements: Vec<F>,
    element_faces: Connectivity,
    face_nodes: Connectivity,
}

impl<C, F> Block<C, F> {
    fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn coordinates(&self) -> &NodalReferenceCoordinates {
        &self.coordinates
    }
    fn elements(&self) -> &[F] {
        &self.elements
    }
    fn element_faces(&self) -> &Connectivity {
        &self.element_faces
    }
    fn face_nodes(&self) -> &Connectivity {
        &self.face_nodes
    }
    fn element_coordinates<const I: usize>(
        coordinates: &Coordinates<I>,
        faces: &[usize],
        face_nodes: &Connectivity,
    ) -> TensorRank1Vec2D<3, I> {
        faces
            .iter()
            .map(|&face| {
                face_nodes[face]
                    .iter()
                    .map(|&node| coordinates[node].clone())
                    .collect()
            })
            .collect()
    }
}

impl<C, F> Debug for Block<C, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ElementBlock {{ constitutive_model: {}, elements: [Virtual; {}] }}",
            type_name::<C>()
                .rsplit("::")
                .next()
                .unwrap()
                .split("<")
                .next()
                .unwrap(),
            self.elements().len()
        )
    }
}

pub trait VirtualElementBlock<C, F>
where
    F: VirtualElement,
    Self: From<(C, NodalReferenceCoordinates, Connectivity, Connectivity)>,
{
}

impl<C, F> From<(C, NodalReferenceCoordinates, Connectivity, Connectivity)> for Block<C, F>
where
    F: VirtualElement,
{
    fn from(
        (constitutive_model, coordinates, element_faces, face_nodes): (
            C,
            NodalReferenceCoordinates,
            Connectivity,
            Connectivity,
        ),
    ) -> Self {
        let elements = element_faces
            .iter()
            .map(|faces| <F>::from(Self::element_coordinates(&coordinates, faces, &face_nodes)))
            .collect();
        Self {
            constitutive_model,
            coordinates,
            elements,
            element_faces,
            face_nodes,
        }
    }
}
