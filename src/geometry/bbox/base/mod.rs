#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, bbox::BoundingBox},
    math::Tensor,
};

impl<const D: usize> BoundingBox<D> {
    pub fn minimum(&self) -> &Coordinate<D> {
        &self.minimum
    }
    pub fn maximum(&self) -> &Coordinate<D> {
        &self.maximum
    }
    pub fn longest_axis(&self) -> usize {
        self.maximum
            .iter()
            .zip(self.minimum.iter())
            .enumerate()
            .map(|(i, (&max, &min))| (i, max - min))
            .max_by(|(_, length_a), (_, length_b)| length_a.partial_cmp(length_b).unwrap())
            .unwrap()
            .0
    }
    pub fn shortest_axis(&self) -> usize {
        self.maximum
            .iter()
            .zip(self.minimum.iter())
            .enumerate()
            .map(|(i, (&max, &min))| (i, max - min))
            .min_by(|(_, length_a), (_, length_b)| length_a.partial_cmp(length_b).unwrap())
            .unwrap()
            .0
    }
}
