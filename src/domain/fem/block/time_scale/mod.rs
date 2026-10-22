use crate::{
    constitutive::solid::elastic::Elastic,
    fem::{
        ElementModelError, NodalCoordinates, NodalReferenceCoordinates,
        block::{
            Block, Densities,
            element::{
                FiniteElementError, mass::LumpedMassFiniteElement,
                solid::elastic::ElasticFiniteElement, time_scale::fastest_time_scale,
            },
        },
        solid::time_scale::TimeScaleElements,
    },
    math::Quantity,
    units::Time,
};

impl<C, F, R, H, const G: usize, const M: usize, const N: usize> TimeScaleElements<3>
    for Block<C, F, G, M, N, R, H>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, M, N> + LumpedMassFiniteElement<G, M, N>,
    R: Densities<G>,
{
    fn fastest_time_scale(
        &self,
        _reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .try_fold(
                Time::seconds(f64::INFINITY),
                |fastest, (element_index, (element, nodes))| {
                    let stiffnesses = element.nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?;
                    let masses = element.nodal_lumped_masses(&self.density().at(element_index));
                    Ok::<_, FiniteElementError>(
                        fastest.min(fastest_time_scale(&stiffnesses, &masses)),
                    )
                },
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
