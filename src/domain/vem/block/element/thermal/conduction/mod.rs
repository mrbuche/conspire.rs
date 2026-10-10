#[cfg(test)]
mod test;

use crate::{
    constitutive::thermal::conduction::ThermalConduction,
    fem::block::element::{
        FiniteElementError, thermal::conduction::ThermalConductionFiniteElement,
    },
    math::{ContractWith, Quantity, Scalar, Tensor, TensorVector},
    units::{Power, PowerPerTemperature, PowerTemperature},
    vem::block::element::{
        Element, VirtualElement, VirtualElementError,
        thermal::{ElementNodalTemperatures, ThermalVirtualElement},
    },
};

pub type ElementNodalForcesThermal = TensorVector<Quantity<Power>>;
pub type ElementNodalStiffnessesThermal = TensorVector<TensorVector<Quantity<PowerPerTemperature>>>;

pub trait ThermalConductionVirtualElement<C>
where
    C: ThermalConduction,
    Self: ThermalVirtualElement,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Result<Quantity<PowerTemperature>, VirtualElementError>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Result<ElementNodalForcesThermal, VirtualElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Result<ElementNodalStiffnessesThermal, VirtualElementError>;
}

/// The nodes that each temperature of the stabilization averages, with their weights.
///
/// The temperatures are those of the nodes, then of the face centers, then of the element
/// center, which the tetrahedra of the stabilization are built on.
fn averages(element: &Element, num_nodes: usize) -> Vec<Vec<(usize, Scalar)>> {
    let center = (0..num_nodes)
        .map(|node| (node, 1.0 / num_nodes as Scalar))
        .collect();
    (0..num_nodes)
        .map(|node| vec![(node, 1.0)])
        .chain(element.faces_nodes().iter().map(|face_nodes| {
            face_nodes
                .iter()
                .map(|&node| (node, 1.0 / face_nodes.len() as Scalar))
                .collect()
        }))
        .chain([center])
        .collect()
}

/// The positions in the [`averages`] of the temperatures of a tetrahedron.
fn reduced(element: &Element, num_nodes: usize, [face, node_b, node_a]: [usize; 3]) -> [usize; 4] {
    [
        num_nodes + face,
        node_b,
        node_a,
        num_nodes + element.faces_nodes().len(),
    ]
}

impl<C> ThermalConductionVirtualElement<C> for Element
where
    C: ThermalConduction,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Result<Quantity<PowerTemperature>, VirtualElementError> {
        let stabilization = self.stabilization();
        let projected = constitutive_model
            .potential(&self.temperature_gradient(nodal_temperatures))
            .map_err(|error| self.upstream(error))?
            * (self.integration_weights()[0] * (1.0 - stabilization));
        self.tetrahedra()
            .iter()
            .zip(self.tetrahedra_temperatures(nodal_temperatures))
            .try_fold(projected, |total, (tetrahedron, temperatures)| {
                Ok::<_, FiniteElementError>(
                    total
                        + ThermalConductionFiniteElement::potential(
                            tetrahedron,
                            constitutive_model,
                            &temperatures,
                        )? * stabilization,
                )
            })
            .map_err(|error| self.upstream(error))
    }
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Result<ElementNodalForcesThermal, VirtualElementError> {
        let num_nodes = nodal_temperatures.len();
        let stabilization = self.stabilization();
        let heat_flux = constitutive_model
            .heat_flux(&self.temperature_gradient(nodal_temperatures))
            .map_err(|error| self.upstream(error))?;
        let weight = self.integration_weights()[0] * (1.0 - stabilization);
        let mut forces = self.gradient_vectors()[0]
            .iter()
            .map(|gradient_vector| -heat_flux.contract_with(gradient_vector) * weight)
            .collect::<ElementNodalForcesThermal>();
        let averages = averages(self, num_nodes);
        let mut reduced_forces = vec![Quantity::<Power>::default(); averages.len()];
        self.tetrahedra()
            .iter()
            .zip(self.tetrahedra_temperatures(nodal_temperatures))
            .zip(self.tetrahedra_nodes())
            .try_for_each(|((tetrahedron, temperatures), &nodes)| {
                let tetrahedron_forces = ThermalConductionFiniteElement::nodal_forces(
                    tetrahedron,
                    constitutive_model,
                    &temperatures,
                )?;
                reduced(self, num_nodes, nodes)
                    .into_iter()
                    .zip(tetrahedron_forces)
                    .for_each(|(index, force)| reduced_forces[index] += force * stabilization);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| self.upstream(error))?;
        reduced_forces
            .iter()
            .zip(averages)
            .for_each(|(&reduced_force, average)| {
                average
                    .iter()
                    .for_each(|&(node, weight)| forces[node] += reduced_force * weight)
            });
        Ok(forces)
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Result<ElementNodalStiffnessesThermal, VirtualElementError> {
        let num_nodes = nodal_temperatures.len();
        let stabilization = self.stabilization();
        let heat_flux_tangent = constitutive_model
            .heat_flux_tangent(&self.temperature_gradient(nodal_temperatures))
            .map_err(|error| self.upstream(error))?;
        let weight = self.integration_weights()[0] * (1.0 - stabilization);
        let gradient_vectors = &self.gradient_vectors()[0];
        let mut stiffnesses = gradient_vectors
            .iter()
            .map(|gradient_vector_a| {
                gradient_vectors
                    .iter()
                    .map(|gradient_vector_b| {
                        -gradient_vector_a.contract_with(&(&heat_flux_tangent * gradient_vector_b))
                            * weight
                    })
                    .collect()
            })
            .collect::<ElementNodalStiffnessesThermal>();
        let averages = averages(self, num_nodes);
        let size = averages.len();
        let mut reduced_stiffnesses =
            vec![vec![Quantity::<PowerPerTemperature>::default(); size]; size];
        self.tetrahedra()
            .iter()
            .zip(self.tetrahedra_temperatures(nodal_temperatures))
            .zip(self.tetrahedra_nodes())
            .try_for_each(|((tetrahedron, temperatures), &nodes)| {
                let tetrahedron_stiffnesses = ThermalConductionFiniteElement::nodal_stiffnesses(
                    tetrahedron,
                    constitutive_model,
                    &temperatures,
                )?;
                let indices = reduced(self, num_nodes, nodes);
                indices.iter().enumerate().for_each(|(a, &row)| {
                    indices.iter().enumerate().for_each(|(b, &column)| {
                        reduced_stiffnesses[row][column] +=
                            tetrahedron_stiffnesses[a][b] * stabilization
                    })
                });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| self.upstream(error))?;
        reduced_stiffnesses
            .iter()
            .zip(&averages)
            .for_each(|(reduced_row, average_row)| {
                reduced_row.iter().zip(&averages).for_each(
                    |(&reduced_stiffness, average_column)| {
                        average_row.iter().for_each(|&(node_a, weight_a)| {
                            average_column.iter().for_each(|&(node_b, weight_b)| {
                                stiffnesses[node_a][node_b] +=
                                    reduced_stiffness * (weight_a * weight_b)
                            })
                        })
                    },
                )
            });
        Ok(stiffnesses)
    }
}
