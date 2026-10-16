#[cfg(test)]
pub mod test;

use crate::{
    constitutive::electrical::dielectric::Dielectric,
    fem::{
        ElementModelError,
        block::{
            Block,
            electrical::{ElectricalElements, NodalPotentials},
            element::{FiniteElementError, electrical::dielectric::DielectricFiniteElement},
        },
        electrical::dielectric::DielectricElements,
    },
    math::Quantity,
    units::Energy,
};

pub use crate::domain::electrical::{NodalForcesElectrical, NodalStiffnessesElectrical};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H> DielectricElements
    for Block<C, F, G, M, N, P, R, H>
where
    C: Dielectric,
    F: DielectricFiniteElement<C, G, M, N, P>,
{
    fn potential(
        &self,
        nodal_potentials: &NodalPotentials,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, element_connectivity)| {
                element.potential(
                    self.constitutive_model(),
                    &self.nodal_potentials_element(element_connectivity, nodal_potentials),
                )
            })
            .sum::<Result<Quantity<Energy>, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_forces_into(
        &self,
        nodal_potentials: &NodalPotentials,
        nodal_forces: &mut NodalForcesElectrical,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &self.nodal_potentials_element(element_connectivity, nodal_potentials),
                    )?
                    .into_iter()
                    .zip(element_connectivity)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_potentials: &NodalPotentials,
        nodal_stiffnesses: &mut NodalStiffnessesElectrical,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &self.nodal_potentials_element(element_connectivity, nodal_potentials),
                    )?
                    .into_iter()
                    .zip(element_connectivity)
                    .for_each(|(object, &node_a)| {
                        object.into_iter().zip(element_connectivity).for_each(
                            |(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            },
                        )
                    });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
