#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef,
        bbox::{BoundingBox, BoundingBoxes},
        mesh::{Connectivity, Mesh},
    },
    math::{Scalar, Tensor},
};

impl<const D: usize> Mesh<D> {
    pub fn bounding_boxes(&self) -> BoundingBoxes<D> {
        self.iter()
            .flatten()
            .map(|nodes| {
                nodes
                    .iter()
                    .map(|&node| &self.coordinates()[node])
                    .collect::<CoordinatesRef<'_, D>>()
                    .into()
            })
            .collect()
    }
    pub fn centroids(&self) -> Coordinates<D> {
        self.iter()
            .flatten()
            .map(|nodes| {
                let count = nodes.len() as Scalar;
                nodes
                    .iter()
                    .map(|&node| &self.coordinates()[node])
                    .sum::<Coordinate<D>>()
                    / count
            })
            .collect()
    }
    pub fn bounding_boxes_and_centroids(
        &self,
    ) -> impl Iterator<Item = (BoundingBox<D>, Coordinate<D>)> {
        self.iter().flatten().map(|nodes| {
            let count = nodes.len() as Scalar;
            (
                nodes
                    .iter()
                    .map(|&node| &self.coordinates()[node])
                    .collect::<CoordinatesRef<'_, D>>()
                    .into(),
                nodes
                    .iter()
                    .map(|&node| &self.coordinates()[node])
                    .sum::<Coordinate<D>>()
                    / count,
            )
        })
    }
    pub fn connectivities(&self) -> &[Connectivity] {
        self.connectivities.members()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Connectivity> {
        self.connectivities.members().iter()
    }
    pub fn coordinates(&self) -> &Coordinates<D> {
        self.coordinates.members()
    }
    pub fn node_element_connectivity(&self) -> &[Vec<usize>] {
        self.nodes_elements.get_or_init(|| {
            let mut nodes_elements = vec![Vec::new(); self.number_of_nodes()];
            let mut element_offset = 0;
            for connectivity in self.iter() {
                let local = connectivity.node_element_connectivity();
                for (node, elems) in local.iter().enumerate() {
                    nodes_elements[node].extend(elems.iter().map(|&e| e + element_offset))
                }
                element_offset += connectivity.number_of_elements();
            }
            nodes_elements
        })
    }
    pub fn node_node_connectivity(&self) -> &[Vec<usize>] {
        self.nodes_nodes.get_or_init(|| {
            let mut nodes_nodes = vec![Vec::new(); self.number_of_nodes()];
            for connectivity in self.iter() {
                connectivity.add_edge_adjacency(&mut nodes_nodes);
            }
            for neighbors in &mut nodes_nodes {
                neighbors.sort_unstable();
                neighbors.dedup();
            }
            nodes_nodes
        })
    }
    pub fn number_of_element_blocks(&self) -> usize {
        self.connectivities().len()
    }
    pub fn number_of_face_blocks(&self) -> Option<usize> {
        let number_of_face_blocks = self
            .iter()
            .filter(|connectivity| connectivity.number_of_faces().is_some())
            .count();
        if number_of_face_blocks > 0 {
            Some(number_of_face_blocks)
        } else {
            None
        }
    }
    pub fn number_of_elements(&self) -> usize {
        self.iter()
            .map(|connectivity| connectivity.number_of_elements())
            .sum()
    }
    pub fn number_of_faces(&self) -> Option<usize> {
        let number_of_faces = self
            .iter()
            .filter_map(|connectivity| connectivity.number_of_faces())
            .sum();
        if number_of_faces > 0 {
            Some(number_of_faces)
        } else {
            None
        }
    }
    pub fn number_of_nodes(&self) -> usize {
        self.coordinates().len()
    }
}
