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

impl<const D: usize, const I: usize, const M: usize, T> From<Mesh<D, I, M, T>>
    for (T, Coordinates<D, I>)
{
    fn from(mesh: Mesh<D, I, M, T>) -> Self {
        (mesh.connectivity, mesh.coordinates)
    }
}
