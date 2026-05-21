#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Coordinates, CoordinatesRef, bbox::BoundingBoxes, mesh::Mesh},
    math::Scalar,
};
use std::iter::ExactSizeIterator;

// impl as BoundingBoxes From<Mesh> etc. instead?

impl<const D: usize, const I: usize, const M: usize, T, U, V> Mesh<D, I, M, T>
where
    for<'a> &'a T: IntoIterator<Item = &'a U>,
    for<'a> &'a U: ExactSizeIterator + IntoIterator<Item = &'a V>,
    V: Copy + Into<usize>,
{
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
                nodes
                    .into_iter()
                    .map(|&node| &self.coordinates[node.into()])
                    .sum::<Coordinate<_, _>>()
                    / nodes.len() as Scalar
            })
            .collect()
    }
    pub fn bounding_boxes_and_centroids(&self) -> (BoundingBoxes<D, I>, Coordinates<D, I>) {
        (&self.connectivity)
            .into_iter()
            .map(|nodes| {
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
                        / nodes.len() as Scalar,
                )
            })
            .unzip()
    }
}
