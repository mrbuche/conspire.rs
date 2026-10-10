#[cfg(test)]
mod test;

use super::{Block, density::Densities, node::mass::NodalMass};
use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{
        ElementModelError, NodalCoordinates, NodalReferenceCoordinates,
        block::element::solid::elastic::ElasticElement,
        solid::time_scale::TimeScaleElements,
        time_scale::{largest_eigenvalue, time_scale_from_eigenvalue},
    },
    math::{Quantity, Scalar},
    units::Time,
};

impl<C, R> TimeScaleElements<3> for Block<C, R>
where
    C: Elastic,
    R: Densities,
{
    fn fastest_time_scale(
        &self,
        _reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.nodes
            .iter()
            .try_fold(0.0_f64, |largest, node| {
                let stiffnesses =
                    node.nodal_stiffnesses(&self.constitutive_model, nodal_coordinates)?;
                let masses: Vec<Scalar> = node
                    .neighbors()
                    .iter()
                    .flat_map(|&neighbor| {
                        let share = self.nodes[neighbor]
                            .nodal_mass(self.density.at(neighbor))
                            .value()
                            / self.nodes[neighbor].neighbors().len() as Scalar;
                        [share; 3]
                    })
                    .collect();
                Ok::<_, crate::constitutive::ConstitutiveError>(largest.max(largest_eigenvalue(
                    masses.len(),
                    |row, column| stiffnesses[row / 3][column / 3][row % 3][column % 3].value(),
                    &masses,
                )))
            })
            .map(time_scale_from_eigenvalue)
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
