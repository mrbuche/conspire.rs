use crate::math::{TensorRank1, TensorRank1List, TensorRank1Vec, Tensor};
use std::ops::Add;

pub type Coordinate<const D: usize, const I: usize> = TensorRank1<D, I>;
pub type Coordinates<const D: usize, const I: usize> = TensorRank1Vec<D, I>;
pub type CoordinatesList<const D: usize, const I: usize, const N: usize> = TensorRank1List<D, I, N>;

pub struct BoundingBox<const D: usize, const I: usize> {
    minimum: Coordinate<D, I>,
    maximum: Coordinate<D, I>,
}

impl<const D: usize, const I: usize> Add for BoundingBox<D, I> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        let mut minimum = self.minimum;
        let mut maximum = self.maximum;
        minimum.iter_mut().zip(other.minimum).for_each(|(min, entry)|
            *min = min.min(entry)
        );
        for i in 0..D {
            // minimum[i] = minimum[i].min(other.minimum[i]);
            maximum[i] = maximum[i].max(other.maximum[i]);
        }
        Self { minimum, maximum }
    }
}
