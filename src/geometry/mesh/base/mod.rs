#[cfg(test)]
mod test;

use crate::math::FxHashMap;
use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef,
        bbox::{BoundingBox, BoundingBoxes},
        mesh::{Connectivity, Mesh, NodeSets, SideSets},
    },
    math::{CrossProduct, Graph, Scalar, Tensor, TensorRank1Vec2D},
};

impl Mesh<3> {
    pub fn normals(&self) -> TensorRank1Vec2D<3, 0> {
        self.iter()
            .map(|connectivity| match connectivity {
                Connectivity::Triangular(triangles) => triangles
                    .iter()
                    .map(|&[node_0, node_1, node_2]| {
                        let u = &self.coordinates()[node_1] - &self.coordinates()[node_0];
                        let v = &self.coordinates()[node_2] - &self.coordinates()[node_0];
                        u.cross(v).normalized()
                    })
                    .collect(),
                _ => panic!(),
            })
            .collect()
    }
}

impl<const D: usize> Mesh<D> {
    pub fn exterior_faces(&self) -> Vec<Vec<usize>> {
        let mut faces = FxHashMap::default();
        self.iter().for_each(|block| {
            let local_faces = block.local_faces();
            block.iter().for_each(|element| {
                local_faces.iter().for_each(|face| {
                    let oriented: Vec<usize> = face.iter().map(|&local| element[local]).collect();
                    let mut key = oriented.clone();
                    key.sort_unstable();
                    faces
                        .entry(key)
                        .and_modify(|(_, count)| *count += 1)
                        .or_insert((oriented, 1));
                })
            })
        });
        faces
            .into_values()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect()
    }
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
    pub fn blocks(&self) -> Option<&[usize]> {
        self.connectivities.numbers()
    }
    pub fn connectivities(&self) -> &[Connectivity] {
        self.connectivities.members()
    }
    pub fn node_sets(&self) -> &[Vec<usize>] {
        self.node_sets.members()
    }
    pub fn node_set_numbers(&self) -> Option<&[usize]> {
        self.node_sets.numbers()
    }
    pub fn set_node_sets(&mut self, node_sets: NodeSets) {
        self.node_sets = node_sets;
    }
    pub fn side_sets(&self) -> &[Vec<(usize, usize)>] {
        self.side_sets.members()
    }
    pub fn side_set_numbers(&self) -> Option<&[usize]> {
        self.side_sets.numbers()
    }
    pub fn set_side_sets(&mut self, side_sets: SideSets) {
        self.side_sets = side_sets;
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
    pub fn node_nodes(&self) -> &Graph {
        self.nodes_nodes.get_or_init(|| {
            let mut nodes_nodes = vec![Vec::new(); self.number_of_nodes()];
            for connectivity in self.iter() {
                connectivity.add_edge_adjacency(&mut nodes_nodes);
            }
            for neighbors in &mut nodes_nodes {
                neighbors.sort_unstable();
                neighbors.dedup();
            }
            nodes_nodes.into()
        })
    }
    pub fn node_node_connectivity(&self) -> &[Vec<usize>] {
        self.node_nodes().adjacency()
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
