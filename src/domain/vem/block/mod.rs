pub mod element;
pub mod solid;

use crate::{
    fem::Elements,
    vem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::element::{ElementNodalCoordinates, VirtualElement},
    },
};
use std::{
    any::type_name,
    fmt::{self, Debug, Formatter},
};

pub type Connectivity = Vec<Vec<usize>>;

pub struct Block<C, F> {
    constitutive_model: C,
    elements: Vec<F>,
    elements_faces: Connectivity,
    elements_nodes: Connectivity,
    faces_nodes: Connectivity,
}

impl<C, F> Block<C, F> {
    fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn elements(&self) -> &[F] {
        &self.elements
    }
    fn element_coordinates<'a>(
        coordinates: &'a NodalCoordinates,
        nodes: &[usize],
    ) -> ElementNodalCoordinates<'a> {
        nodes.iter().map(|&node| &coordinates[node]).collect()
    }
    pub fn elements_faces(&self) -> &[Vec<usize>] {
        &self.elements_faces
    }
    fn elements_nodes(&self) -> &[Vec<usize>] {
        &self.elements_nodes
    }
    pub fn faces_nodes(&self) -> &[Vec<usize>] {
        &self.faces_nodes
    }
}

impl<C, F> Debug for Block<C, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Block {{ constitutive model: {}, elements: [Virtual; {}] }}",
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

impl<C, F> Elements for Block<C, F> {
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        self.elements_nodes().iter().for_each(|nodes| {
            nodes.iter().for_each(|&node_a| {
                nodes
                    .iter()
                    .for_each(|&node_b| neighbors[node_a].push(node_b))
            })
        })
    }
}

pub trait VirtualElements<C, F>
where
    F: VirtualElement,
    Self: for<'a> From<(C, Connectivity, Connectivity, &'a NodalReferenceCoordinates)>,
{
}

impl<C, F> From<(C, Connectivity, Connectivity, &NodalReferenceCoordinates)> for Block<C, F>
where
    F: VirtualElement,
{
    fn from(
        (constitutive_model, elements_faces, faces_nodes, coordinates): (
            C,
            Connectivity,
            Connectivity,
            &NodalReferenceCoordinates,
        ),
    ) -> Self {
        let (elements, elements_nodes) = elements_faces
            .iter()
            .map(|element_faces| {
                let element_coordinates = element_faces
                    .iter()
                    .map(|&face| {
                        faces_nodes[face]
                            .iter()
                            .map(|&node| coordinates[node].clone())
                            .collect()
                    })
                    .collect();
                let mut element_nodes = element_faces
                    .iter()
                    .flat_map(|&face| faces_nodes[face].clone())
                    .collect::<Vec<_>>();
                element_nodes.sort();
                element_nodes.dedup();
                (
                    <F>::from((
                        element_coordinates,
                        element_faces,
                        &element_nodes,
                        &faces_nodes,
                    )),
                    element_nodes,
                )
            })
            .unzip();
        Self {
            constitutive_model,
            elements,
            elements_faces,
            elements_nodes,
            faces_nodes,
        }
    }
}
