use super::Node;
use crate::{
    math::Quantity,
    units::{Density, Mass},
};

pub trait NodalMass {
    fn nodal_mass(&self, density: Quantity<Density>) -> Quantity<Mass>;
}

impl NodalMass for Node {
    fn nodal_mass(&self, density: Quantity<Density>) -> Quantity<Mass> {
        density * self.volume
    }
}
