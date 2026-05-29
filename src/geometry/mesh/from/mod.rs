#[cfg(test)]
pub mod test;

use crate::geometry::{
    Coordinates,
    mesh::{Connectivities, Mesh},
};

impl<const D: usize, T> From<(Connectivities<T>, Coordinates<D>)> for Mesh<D, T> {
    fn from((connectivities, coordinates): (Connectivities<T>, Coordinates<D>)) -> Self {
        Self {
            connectivities,
            coordinates,
        }
    }
}

impl<const D: usize, T> From<(Connectivities<T>, &Coordinates<D>)> for Mesh<D, T> {
    fn from((connectivities, coordinates): (Connectivities<T>, &Coordinates<D>)) -> Self {
        Self {
            connectivities,
            coordinates: coordinates.clone(),
        }
    }
}
