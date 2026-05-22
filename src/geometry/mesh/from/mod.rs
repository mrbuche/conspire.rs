#[cfg(test)]
mod test;

use crate::geometry::{Coordinates, Mesh};

impl<const D: usize, const I: usize, const M: usize, T> From<(T, Coordinates<D, I>)>
    for Mesh<D, I, M, T>
{
    fn from((connectivity, coordinates): (T, Coordinates<D, I>)) -> Self {
        Self {
            coordinates,
            connectivity,
        }
    }
}

impl<const D: usize, const I: usize, const M: usize, T> From<(T, &Coordinates<D, I>)>
    for Mesh<D, I, M, T>
{
    fn from((connectivity, coordinates): (T, &Coordinates<D, I>)) -> Self {
        Self {
            coordinates: coordinates.clone(),
            connectivity,
        }
    }
}

impl<const D: usize, const I: usize, const M: usize, T> From<(&T, Coordinates<D, I>)>
    for Mesh<D, I, M, T>
where
    T: Clone,
{
    fn from((connectivity, coordinates): (&T, Coordinates<D, I>)) -> Self {
        Self {
            coordinates,
            connectivity: connectivity.clone(),
        }
    }
}

impl<const D: usize, const I: usize, const M: usize, T> From<(&T, &Coordinates<D, I>)>
    for Mesh<D, I, M, T>
where
    T: Clone,
{
    fn from((connectivity, coordinates): (&T, &Coordinates<D, I>)) -> Self {
        Self {
            coordinates: coordinates.clone(),
            connectivity: connectivity.clone(),
        }
    }
}
