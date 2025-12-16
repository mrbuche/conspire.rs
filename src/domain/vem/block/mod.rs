pub mod element;

use crate::vem::{NodalReferenceCoordinates, block::element::VirtualElement};
use std::{
    any::type_name,
    fmt::{self, Debug, Formatter},
};

pub type Connectivity = Vec<Vec<usize>>;

pub struct ElementBlock<C, F> {
    constitutive_model: C,
    coordinates: NodalReferenceCoordinates,
    elements: Vec<F>,
    element_faces: Connectivity,
    face_nodes: Connectivity,
}

impl<C, F> ElementBlock<C, F> {
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
}

impl<C, F> Debug for ElementBlock<C, F> {
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
{
    fn new(
        constitutive_model: C,
        coordinates: NodalReferenceCoordinates,
        element_faces: Connectivity,
        face_nodes: Connectivity,
    ) -> Self;
}

impl<C, F> VirtualElementBlock<C, F> for ElementBlock<C, F>
where
    F: VirtualElement,
{
    fn new(
        constitutive_model: C,
        coordinates: NodalReferenceCoordinates,
        element_faces: Connectivity,
        face_nodes: Connectivity,
    ) -> Self {
        let elements = element_faces
            .iter()
            .map(|faces| {
                <F>::from(
                    faces
                        .iter()
                        .map(|&face| {
                            face_nodes[face]
                                .iter()
                                .map(|&node| coordinates[node].clone())
                                .collect()
                        })
                        .collect(),
                )
            })
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
