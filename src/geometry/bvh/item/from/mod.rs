#[cfg(test)]
mod test;

use crate::geometry::{BoundingBox, BoundingBoxUnite, bvh::item::Item};

impl<const D: usize, const I: usize, T> From<&[Item<D, I, T>]> for BoundingBox<D, I> {
    fn from(items: &[Item<D, I, T>]) -> Self {
        items
            .iter()
            .skip(1)
            .fold(items[0].bounding_box.clone(), |bbox, item| {
                bbox.unite(&item.bounding_box)
            })
    }
}
