#[cfg(test)]
pub mod test;

use crate::geometry::{Coordinates, Mesh};

impl<const D: usize, const M: usize, T> From<(T, Coordinates<D>)> for Mesh<D, M, T> {
    fn from((connectivity, coordinates): (T, Coordinates<D>)) -> Self {
        Self {
            coordinates,
            connectivity,
        }
    }
}

impl<const D: usize, const M: usize, T> From<(T, &Coordinates<D>)> for Mesh<D, M, T> {
    fn from((connectivity, coordinates): (T, &Coordinates<D>)) -> Self {
        Self {
            coordinates: coordinates.clone(),
            connectivity,
        }
    }
}

impl<const D: usize, const M: usize, T> From<(&T, Coordinates<D>)> for Mesh<D, M, T>
where
    T: Clone,
{
    fn from((connectivity, coordinates): (&T, Coordinates<D>)) -> Self {
        Self {
            coordinates,
            connectivity: connectivity.clone(),
        }
    }
}

impl<const D: usize, const M: usize, T> From<(&T, &Coordinates<D>)> for Mesh<D, M, T>
where
    T: Clone,
{
    fn from((connectivity, coordinates): (&T, &Coordinates<D>)) -> Self {
        Self {
            coordinates: coordinates.clone(),
            connectivity: connectivity.clone(),
        }
    }
}
