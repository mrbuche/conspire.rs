#[cfg(test)]
mod test;

use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{ElementModelError, time_scale::TimeScaleElements},
    math::Quantity,
    units::Time,
    vem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::{
            Block, Densities,
            element::{
                VirtualElementError, mass::LumpedMassVirtualElement,
                solid::elastic::ElasticVirtualElement, time_scale::fastest_time_scale,
            },
        },
    },
};

impl<C, F, R> TimeScaleElements<3> for Block<C, F, R>
where
    C: Elastic,
    F: ElasticVirtualElement<C> + LumpedMassVirtualElement,
    R: Densities,
{
    fn fastest_time_scale(
        &self,
        reference_coordinates: &NodalReferenceCoordinates,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .enumerate()
            .try_fold(
                Time::seconds(f64::INFINITY),
                |fastest, (element_index, (element, nodes))| {
                    let stiffnesses = element.nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?;
                    let masses = element.nodal_lumped_masses(
                        self.density().at(element_index),
                        &Self::element_coordinates(reference_coordinates, nodes),
                    );
                    Ok::<_, VirtualElementError>(
                        fastest.min(fastest_time_scale(&stiffnesses, &masses)),
                    )
                },
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
