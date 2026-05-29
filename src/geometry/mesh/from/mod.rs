#[cfg(test)]
pub mod test;

use crate::geometry::{
    Coordinates,
    mesh::{Connectivities, Mesh},
};

impl<const D: usize> From<(Connectivities, Coordinates<D>)> for Mesh<D> {
    fn from((connectivities, coordinates): (Connectivities, Coordinates<D>)) -> Self {
        Self {
            connectivities,
            coordinates,
        }
    }
}

impl<const D: usize> From<(Connectivities, &Coordinates<D>)> for Mesh<D> {
    fn from((connectivities, coordinates): (Connectivities, &Coordinates<D>)) -> Self {
        Self {
            connectivities,
            coordinates: coordinates.clone(),
        }
    }
}
