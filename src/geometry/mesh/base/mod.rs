#[cfg(test)]
mod test;
use crate::math::Reference;

use crate::math::FxHashMap;
use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef,
        bbox::{BoundingBox, BoundingBoxes},
        mesh::{Connectivity, Mesh, NodeSet, NodeSets, SideSet, SideSets},
    },
    math::{CrossProduct, Incidence, Labels, Scalar, Tensor, TensorRank1Vec2D},
    units::Dimensionless,
};

impl Mesh<3> {
    pub fn normals(&self) -> TensorRank1Vec2D<3, Reference, Dimensionless> {
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
    /// The nodes of the faces in a side set, keeping its label and name.
    ///
    /// # Panics
    /// Panics if the side set references an element or face the mesh does not have.
    pub fn nodes_from(&self, side_set: &SideSet) -> NodeSet {
        let mut ordinals = FxHashMap::<usize, Vec<usize>>::default();
        side_set
            .sides()
            .iter()
            .for_each(|&(element, ordinal)| ordinals.entry(element).or_default().push(ordinal));
        let mut nodes = Vec::new();
        let mut index = 0;
        let mut found = 0;
        for block in self.iter() {
            for element in block.iter() {
                if let Some(ordinals) = ordinals.get(&index) {
                    let faces = block.element_faces(element);
                    ordinals
                        .iter()
                        .for_each(|&ordinal| nodes.extend_from_slice(&faces[ordinal]));
                    found += 1;
                }
                index += 1;
            }
        }
        assert_eq!(
            found,
            ordinals.len(),
            "The side set references an element the mesh does not have."
        );
        nodes.sort_unstable();
        nodes.dedup();
        let mut node_set = NodeSet::new(nodes);
        if let Some(label) = side_set.label() {
            node_set = node_set.with_label(label);
        }
        if let Some(name) = side_set.name() {
            node_set = node_set.with_name(name);
        }
        node_set
    }
    /// The nodes whose coordinates satisfy a predicate.
    pub fn select_nodes(&self, predicate: impl Fn(&Coordinate<D>) -> bool) -> NodeSet {
        NodeSet::new(
            self.coordinates
                .iter()
                .enumerate()
                .filter_map(|(node, coordinate)| predicate(coordinate).then_some(node))
                .collect(),
        )
    }
    pub fn exterior_faces(&self) -> Vec<Vec<usize>> {
        let mut faces = FxHashMap::default();
        self.iter().for_each(|block| {
            block.iter().for_each(|element| {
                block
                    .element_faces(element)
                    .into_iter()
                    .for_each(|oriented| {
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
            .flat_map(|block| {
                block
                    .iter()
                    .map(move |element| block.element_nodes(element))
            })
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
            .flat_map(|block| {
                block
                    .iter()
                    .map(move |element| block.element_nodes(element))
            })
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
        self.iter()
            .flat_map(|block| {
                block
                    .iter()
                    .map(move |element| block.element_nodes(element))
            })
            .map(|nodes| {
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
    pub fn block_labels(&self) -> Option<&[usize]> {
        self.connectivities.labels()
    }
    pub fn connectivities(&self) -> &[Connectivity] {
        self.connectivities.blocks()
    }
    pub fn node_sets(&self) -> &NodeSets {
        &self.node_sets
    }
    pub fn set_node_sets(&mut self, node_sets: NodeSets) {
        self.node_sets = node_sets;
    }
    pub fn side_sets(&self) -> &SideSets {
        &self.side_sets
    }
    pub fn set_side_sets(&mut self, side_sets: SideSets) {
        self.side_sets = side_sets;
    }
    pub fn iter(&self) -> impl Iterator<Item = &Connectivity> {
        self.connectivities.blocks().iter()
    }
    pub fn node_labels(&self) -> Option<&[usize]> {
        self.node_labels.as_ref().map(Labels::as_slice)
    }
    pub fn coordinates(&self) -> &Coordinates<D> {
        &self.coordinates
    }
    pub fn node_element_connectivity(&self) -> &[Vec<usize>] {
        self.nodes_elements
            .get_or_init(|| {
                let number_of_nodes = self.number_of_nodes();
                let mut nodes_elements = vec![Vec::new(); number_of_nodes];
                let mut element_offset = 0;
                for connectivity in self.iter() {
                    connectivity
                        .node_element_connectivity(number_of_nodes)
                        .as_slice()
                        .iter()
                        .zip(&mut nodes_elements)
                        .for_each(|(elements, node_elements)| {
                            node_elements.extend(elements.iter().map(|&e| e + element_offset))
                        });
                    element_offset += connectivity.number_of_elements();
                }
                Incidence::new(nodes_elements, element_offset)
            })
            .as_slice()
    }
    pub fn node_nodes(&self) -> &Incidence {
        self.nodes_nodes.get_or_init(|| {
            let mut nodes_nodes = vec![Vec::new(); self.number_of_nodes()];
            for connectivity in self.iter() {
                connectivity.add_edge_adjacency(&mut nodes_nodes);
            }
            for neighbors in &mut nodes_nodes {
                neighbors.sort_unstable();
                neighbors.dedup();
            }
            Incidence::adjacency(nodes_nodes)
        })
    }
    pub fn node_node_connectivity(&self) -> &[Vec<usize>] {
        self.node_nodes().as_slice()
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
