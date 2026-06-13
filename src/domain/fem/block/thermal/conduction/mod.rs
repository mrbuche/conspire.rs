#[cfg(test)]
pub mod test;

use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    fem::{
        ElementModelError,
        block::{
            Block,
            element::{FiniteElementError, thermal::conduction::ThermalConductionFiniteElement},
            thermal::{NodalTemperatures, ThermalElements},
        },
        thermal::conduction::ThermalConductionElements,
    },
    math::{Scalar, SquareMatrix, Vector},
};

pub type NodalForcesThermal = Vector;
pub type NodalStiffnessesThermal = SquareMatrix;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> ThermalConductionElements
    for Block<C, F, G, M, N, P>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P>,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, element_connectivity)| {
                element.potential(
                    self.constitutive_model(),
                    &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                )
            })
            .sum()
        {
            Ok(potential) => Ok(potential),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_forces_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_forces: &mut NodalForcesThermal,
    ) -> Result<(), ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
                    )?
                    .into_iter()
                    .zip(element_connectivity)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_temperatures: &NodalTemperatures,
        nodal_stiffnesses: &mut NodalStiffnessesThermal,
    ) -> Result<(), ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .try_for_each(|(element, element_connectivity)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &self.nodal_temperatures_element(element_connectivity, nodal_temperatures),
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
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
