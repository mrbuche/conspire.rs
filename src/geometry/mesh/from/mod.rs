#[cfg(test)]
pub mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh},
    },
    math::Set,
};

impl<const D: usize> From<(Vec<Connectivity>, Coordinates<D>)> for Mesh<D> {
    fn from((connectivities, coordinates): (Vec<Connectivity>, Coordinates<D>)) -> Self {
        Self {
            connectivities: Connectivities::from(connectivities),
            coordinates: Set::from(coordinates),
        }
    }
}

impl<const D: usize> From<(Vec<Connectivity>, &Coordinates<D>)> for Mesh<D> {
    fn from((connectivities, coordinates): (Vec<Connectivity>, &Coordinates<D>)) -> Self {
        Self {
            connectivities: Connectivities::from(connectivities),
            coordinates: Set::from(coordinates.clone()),
        }
    }
}
