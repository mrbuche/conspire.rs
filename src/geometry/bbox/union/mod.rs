#[cfg(test)]
mod test;

use crate::{
    geometry::bbox::{BoundingBox, Union},
    math::Tensor,
};

impl<const D: usize, const I: usize> Union<Self> for BoundingBox<D, I> {
    type Output = Self;
    fn union(self, other: Self) -> Self::Output {
        let mut minimum = self.minimum;
        let mut maximum = self.maximum;
        minimum
            .iter_mut()
            .zip(other.minimum)
            .zip(maximum.iter_mut().zip(other.maximum))
            .for_each(|((min, other_min), (max, other_max))| {
                *min = min.min(other_min);
                *max = max.max(other_max);
            });
        Self { minimum, maximum }
    }
}

impl<const D: usize, const I: usize> Union<BoundingBox<D, I>> for &BoundingBox<D, I> {
    type Output = BoundingBox<D, I>;
    fn union(self, other: BoundingBox<D, I>) -> Self::Output {
        let mut minimum = self.minimum.clone();
        let mut maximum = self.maximum.clone();
        minimum
            .iter_mut()
            .zip(other.minimum)
            .zip(maximum.iter_mut().zip(other.maximum))
            .for_each(|((min, other_min), (max, other_max))| {
                *min = min.min(other_min);
                *max = max.max(other_max);
            });
        BoundingBox { minimum, maximum }
    }
}

impl<const D: usize, const I: usize> Union<&Self> for BoundingBox<D, I> {
    type Output = Self;
    fn union(self, other: &Self) -> Self::Output {
        let mut minimum = self.minimum;
        let mut maximum = self.maximum;
        minimum
            .iter_mut()
            .zip(other.minimum.iter())
            .zip(maximum.iter_mut().zip(other.maximum.iter()))
            .for_each(|((min, &other_min), (max, &other_max))| {
                *min = min.min(other_min);
                *max = max.max(other_max);
            });
        Self { minimum, maximum }
    }
}

impl<const D: usize, const I: usize> Union<Self> for &BoundingBox<D, I> {
    type Output = BoundingBox<D, I>;
    fn union(self, other: Self) -> Self::Output {
        let mut minimum = self.minimum.clone();
        let mut maximum = self.maximum.clone();
        minimum
            .iter_mut()
            .zip(other.minimum.iter())
            .zip(maximum.iter_mut().zip(other.maximum.iter()))
            .for_each(|((min, &other_min), (max, &other_max))| {
                *min = min.min(other_min);
                *max = max.max(other_max);
            });
        BoundingBox { minimum, maximum }
    }
}
