#[cfg(test)]
pub mod test;

use crate::{
    geometry::bbox::{BoundingBox, Unite},
    math::Tensor,
};

impl<const D: usize> Unite<Self> for BoundingBox<D> {
    type Output = Self;
    fn unite(self, other: Self) -> Self::Output {
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

impl<const D: usize> Unite<BoundingBox<D>> for &BoundingBox<D> {
    type Output = BoundingBox<D>;
    fn unite(self, other: BoundingBox<D>) -> Self::Output {
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

impl<const D: usize> Unite<&Self> for BoundingBox<D> {
    type Output = Self;
    fn unite(self, other: &Self) -> Self::Output {
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

impl<const D: usize> Unite<Self> for &BoundingBox<D> {
    type Output = BoundingBox<D>;
    fn unite(self, other: Self) -> Self::Output {
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
