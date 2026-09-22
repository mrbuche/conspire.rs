pub mod element;
pub mod solid;
#[cfg(test)]
mod test;

use crate::{
    domain::block::{element::Elements, feti::LocalSupport},
    geometry::mesh::PolytopalConnectivity,
    math::TensorRank1Vec,
    vem::{NodalReferenceCoordinates, block::element::VirtualElement},
};
use std::fmt::{self, Debug, Formatter};

pub struct Block<C, F> {
    constitutive_model: C,
    connectivity: PolytopalConnectivity<3>,
    elements: Vec<F>,
    elements_nodes: Vec<Vec<usize>>,
}

impl<C, F> LocalSupport for Block<C, F> {}

impl<C, F> Block<C, F> {
    fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn elements(&self) -> &[F] {
        &self.elements
    }
    fn element_coordinates<I, U>(
        coordinates: &TensorRank1Vec<3, I, U>,
        nodes: &[usize],
    ) -> TensorRank1Vec<3, I, U> {
        nodes
            .iter()
            .map(|&node| coordinates[node].clone())
            .collect()
    }
    pub fn elements_faces(&self) -> &[Vec<usize>] {
        self.connectivity.elements_faces()
    }
    fn elements_nodes(&self) -> &[Vec<usize>] {
        &self.elements_nodes
    }
    pub fn faces_nodes(&self) -> &[Vec<usize>] {
        self.connectivity.faces_nodes()
    }
}

impl<C, F> Debug for Block<C, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Block {{ constitutive model: {}, elements: [Virtual; {}] }}",
            crate::domain::block::trimmed_type_name::<C>(),
            self.elements().len()
        )
    }
}

impl<C, F> Elements for Block<C, F> {
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        crate::domain::block::add_node_neighbors(
            self.elements_nodes().iter().map(|nodes| nodes.as_slice()),
            neighbors,
        )
    }
}

impl<C, F> From<(C, PolytopalConnectivity<3>, &NodalReferenceCoordinates)> for Block<C, F>
where
    F: VirtualElement,
{
    fn from(
        (constitutive_model, connectivity, coordinates): (
            C,
            PolytopalConnectivity<3>,
            &NodalReferenceCoordinates,
        ),
    ) -> Self {
        let faces_nodes = connectivity.faces_nodes();
        let (elements, elements_nodes) = connectivity
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
                        faces_nodes,
                    )),
                    element_nodes,
                )
            })
            .unzip();
        Self {
            constitutive_model,
            connectivity,
            elements,
            elements_nodes,
        }
    }
}

impl<C, F>
    From<(
        C,
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
        &NodalReferenceCoordinates,
    )> for Block<C, F>
where
    F: VirtualElement,
{
    fn from(
        (constitutive_model, elements_faces, faces_nodes, coordinates): (
            C,
            Vec<Vec<usize>>,
            Vec<Vec<usize>>,
            &NodalReferenceCoordinates,
        ),
    ) -> Self {
        Self::from((
            constitutive_model,
            PolytopalConnectivity::from((elements_faces, faces_nodes)),
            coordinates,
        ))
    }
}
