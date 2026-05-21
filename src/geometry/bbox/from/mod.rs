#[cfg(test)]
mod test;

use crate::geometry::{CoordinateList, Coordinates, CoordinatesRef, bbox::BoundingBox};

impl<const D: usize, const I: usize> From<Coordinates<D, I>> for BoundingBox<D, I> {
    fn from(coordinates: Coordinates<D, I>) -> Self {
        let [minimum, maximum] = coordinates.bounding_box().into();
        Self { minimum, maximum }
    }
}

impl<'a, const D: usize, const I: usize> From<CoordinatesRef<'a, D, I>> for BoundingBox<D, I> {
    fn from(coordinates: CoordinatesRef<'a, D, I>) -> Self {
        let [minimum, maximum] = coordinates.bounding_box().into();
        Self { minimum, maximum }
    }
}

impl<const D: usize, const I: usize, const N: usize> From<CoordinateList<D, I, N>>
    for BoundingBox<D, I>
{
    fn from(coordinates: CoordinateList<D, I, N>) -> Self {
        let [minimum, maximum] = coordinates.bounding_box().into();
        Self { minimum, maximum }
    }
}
