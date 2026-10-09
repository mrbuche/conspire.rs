use super::{Block, density::Densities, node::mass::NodalMass};
use crate::{
    domain::{
        NodalReferenceCoordinates,
        solid::mass::{LumpedMassElements, NodalLumpedMasses},
    },
    math::Quantity,
    units::Mass,
};

impl<C, R> LumpedMassElements<3> for Block<C, R>
where
    R: Densities,
{
    fn nodal_lumped_masses_into(
        &self,
        _reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_lumped_masses: &mut NodalLumpedMasses,
    ) {
        self.nodes.iter().enumerate().for_each(|(index, node)| {
            nodal_lumped_masses[index] += node.nodal_mass(self.density.at(index))
        })
    }
}

impl<C, R> Block<C, R>
where
    R: Densities,
{
    pub fn mass(&self) -> Quantity<Mass> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(index, node)| node.nodal_mass(self.density.at(index)))
            .sum()
    }
}
