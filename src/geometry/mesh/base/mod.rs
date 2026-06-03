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
                    .map(|&node| &self.coordinates[node])
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
                    .map(|&node| &self.coordinates[node])
                    .sum::<Coordinate<D>>()
                    / count
            })
            .collect()
    }
    pub fn bounding_boxes_and_centroids(
        &self,
    ) -> impl Iterator<Item = (BoundingBox<D>, Coordinate<D>)> + '_ {
        self.iter().flatten().map(|nodes| {
            let count = nodes.len() as Scalar;
            (
                nodes
                    .iter()
                    .map(|&node| &self.coordinates[node])
                    .collect::<CoordinatesRef<'_, D>>()
                    .into(),
                nodes
                    .iter()
                    .map(|&node| &self.coordinates[node])
                    .sum::<Coordinate<D>>()
                    / count,
            )
        })
    }
    pub fn connectivities(&self) -> &[Connectivity] {
        self.connectivities.members()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Connectivity> + '_ {
        self.connectivities.members().iter()
    }
    pub fn coordinates(&self) -> &Coordinates<D> {
        &self.coordinates
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
        self.coordinates.len()
    }
}
