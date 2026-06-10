#[cfg(test)]
pub mod test;

use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    fem::{
        FiniteElementModelError,
        block::{
            Block,
            element::{FiniteElementError, thermal::conduction::ThermalConductionFiniteElement},
            thermal::{NodalTemperatures, ThermalFiniteElementBlock},
        },
        thermal::conduction::ThermalConductionFiniteElements,
    },
    math::{Scalar, SquareMatrix, Tensor, Vector},
};

pub type NodalForcesThermal = Vector;
pub type NodalStiffnessesThermal = SquareMatrix;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ThermalConductionFiniteElements for Block<C, F, G, M, N, P>
where
    C: ThermalConduction,
    F: ThermalConductionFiniteElement<C, G, M, N, P>,
{
    fn potential(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<Scalar, FiniteElementModelError> {
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
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_forces(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalForcesThermal, FiniteElementModelError> {
        let mut nodal_forces = NodalForcesThermal::zero(nodal_temperatures.len());
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
            Ok(()) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_temperatures: &NodalTemperatures,
    ) -> Result<NodalStiffnessesThermal, FiniteElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesThermal::zero(nodal_temperatures.len());
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
            Ok(()) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
