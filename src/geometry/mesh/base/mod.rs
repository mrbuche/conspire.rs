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
        self.connectivities
            .iter()
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
        self.connectivities
            .iter()
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
        self.connectivities.iter().flatten().map(|nodes| {
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
        &self.connectivities
    }
    pub fn coordinates(&self) -> &Coordinates<D> {
        &self.coordinates
    }
    pub fn number_of_blocks(&self) -> usize {
        self.connectivities.len()
    }
    pub fn number_of_elements(&self) -> usize {
        self.connectivities
            .iter()
            .map(|connectivity| connectivity.len())
            .sum()
    }
    pub fn number_of_nodes(&self) -> usize {
        self.coordinates.len()
    }
}
