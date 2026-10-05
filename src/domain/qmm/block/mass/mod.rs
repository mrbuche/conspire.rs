#[cfg(test)]
mod test;

use super::Block;
use crate::{
    domain::mass::{ConsistentMassElements, NodalMasses},
    math::Quantity,
    units::Density,
};

impl<C> ConsistentMassElements for Block<C, Quantity<Density>> {
    fn nodal_masses_into(&self, nodal_masses: &mut NodalMasses) {
        self.inner_products
            .iter()
            .enumerate()
            .for_each(|(node_a, row)| {
                row.iter().for_each(|&(node_b, inner_product)| {
                    nodal_masses[node_a][node_b] += self.density * inner_product
                })
            })
    }
}
