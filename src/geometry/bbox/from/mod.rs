#[cfg(test)]
mod test;

use crate::geometry::{CoordinateList, Coordinates, CoordinatesRef, bbox::BoundingBox};

impl<const D: usize> From<Coordinates<D>> for BoundingBox<D> {
    fn from(coordinates: Coordinates<D>) -> Self {
        let [minimum, maximum] = coordinates.bounding_box().into();
        Self { minimum, maximum }
    }
}

impl<'a, const D: usize> From<CoordinatesRef<'a, D>> for BoundingBox<D> {
    fn from(coordinates: CoordinatesRef<'a, D>) -> Self {
        let [minimum, maximum] = coordinates.bounding_box().into();
        Self { minimum, maximum }
    }
}

impl<const D: usize, const N: usize> From<CoordinateList<D, N>> for BoundingBox<D> {
    fn from(coordinates: CoordinateList<D, N>) -> Self {
        let [minimum, maximum] = coordinates.bounding_box().into();
        Self { minimum, maximum }
    }
}
