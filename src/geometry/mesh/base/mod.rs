#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef,
        bbox::{BoundingBox, BoundingBoxes},
        mesh::Mesh,
    },
    math::{Scalar, Tensor},
};

impl<const D: usize, const I: usize, const M: usize, T, U, V> Mesh<D, I, M, T>
where
    for<'a> &'a T: IntoIterator<Item = &'a U>,
    for<'a> &'a U: IntoIterator<Item = &'a V>,
    V: Copy + Into<usize>,
{
    pub fn connectivity(&self) -> &T {
        &self.connectivity
    }
    pub fn coordinates(&self) -> &Coordinates<D, I> {
        &self.coordinates
    }
    pub fn number_of_nodes(&self) -> usize {
        self.coordinates.len()
    }
    pub fn bounding_boxes(&self) -> BoundingBoxes<D, I> {
        (&self.connectivity)
            .into_iter()
            .map(|nodes| {
                nodes
                    .into_iter()
                    .map(|&node| &self.coordinates[node.into()])
                    .collect::<CoordinatesRef<'_, _, _>>()
                    .into()
            })
            .collect()
    }
    pub fn centroids(&self) -> Coordinates<D, I> {
        (&self.connectivity)
            .into_iter()
            .map(|nodes| {
                let count = nodes.into_iter().count() as Scalar;
                nodes
                    .into_iter()
                    .map(|&node| &self.coordinates[node.into()])
                    .sum::<Coordinate<_, _>>()
                    / count
            })
            .collect()
    }
    pub fn bounding_boxes_and_centroids(
        &self,
    ) -> impl Iterator<Item = (BoundingBox<D, I>, Coordinate<D, I>)> {
        (&self.connectivity).into_iter().map(|nodes| {
            let count = nodes.into_iter().count() as Scalar;
            (
                nodes
                    .into_iter()
                    .map(|&node| &self.coordinates[node.into()])
                    .collect::<CoordinatesRef<'_, _, _>>()
                    .into(),
                nodes
                    .into_iter()
                    .map(|&node| &self.coordinates[node.into()])
                    .sum::<Coordinate<_, _>>()
                    / count,
            )
        })
    }
}
