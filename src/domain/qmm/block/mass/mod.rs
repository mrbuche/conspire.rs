#[cfg(test)]
mod test;

use super::Block;
use crate::{
    domain::mass::{ConsistentMassElements, LumpedMassElements, NodalLumpedMasses, NodalMasses},
    math::{Quantity, Tensor},
    units::{Density, Volume},
};

impl<C> LumpedMassElements for Block<C, Quantity<Density>> {
    fn nodal_lumped_masses_into(&self, nodal_lumped_masses: &mut NodalLumpedMasses) {
        let diagonal = |node: usize| {
            self.inner_products[node]
                .iter()
                .find(|&&(other, _)| other == node)
                .map_or(0.0, |(_, inner_product)| inner_product.value())
        };
        let total: f64 = self
            .inner_products
            .iter()
            .flatten()
            .map(|(_, inner_product)| inner_product.value())
            .sum();
        let trace: f64 = (0..self.inner_products.len()).map(diagonal).sum();
        nodal_lumped_masses
            .iter_mut()
            .enumerate()
            .for_each(|(node, nodal_lumped_mass)| {
                *nodal_lumped_mass +=
                    self.density * Quantity::<Volume>::new(diagonal(node) * total / trace)
            })
    }
}

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
