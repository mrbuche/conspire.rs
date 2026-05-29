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
        todo!()
        // (&self.connectivity)
        //     .into_iter()
        //     .map(|nodes| {
        //         nodes
        //             .into_iter()
        //             .map(|&node| &self.coordinates[node.into()])
        //             .collect::<CoordinatesRef<'_, _>>()
        //             .into()
        //     })
        //     .collect()
    }
    pub fn centroids(&self) -> Coordinates<D> {
        todo!()
        // (&self.connectivity)
        //     .into_iter()
        //     .map(|nodes| {
        //         let count = nodes.into_iter().count() as Scalar;
        //         nodes
        //             .into_iter()
        //             .map(|&node| &self.coordinates[node.into()])
        //             .sum::<Coordinate<_>>()
        //             / count
        //     })
        //     .collect()
    }
    // pub fn bounding_boxes_and_centroids(
    //     &self,
    // ) -> impl Iterator<Item = (BoundingBox<D>, Coordinate<D>)> {
    //     todo!()
    //     // (&self.connectivity).into_iter().map(|nodes| {
    //     //     let count = nodes.into_iter().count() as Scalar;
    //     //     (
    //     //         nodes
    //     //             .into_iter()
    //     //             .map(|&node| &self.coordinates[node.into()])
    //     //             .collect::<CoordinatesRef<'_, _>>()
    //     //             .into(),
    //     //         nodes
    //     //             .into_iter()
    //     //             .map(|&node| &self.coordinates[node.into()])
    //     //             .sum::<Coordinate<_>>()
    //     //             / count,
    //     //     )
    //     // })
    // }
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
