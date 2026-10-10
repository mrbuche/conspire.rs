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
        let stabilization = self.stabilization();
        let inverse_num_nodes = 1.0 / nodal_temperatures.len() as Scalar;
        let heat_flux = constitutive_model
            .heat_flux(&self.temperature_gradient(nodal_temperatures))
            .map_err(|error| self.upstream(error))?;
        let weight = self.integration_weights()[0] * (1.0 - stabilization);
        let mut forces = self.gradient_vectors()[0]
            .iter()
            .map(|gradient_vector| -heat_flux.contract_with(gradient_vector) * weight)
            .collect::<ElementNodalForcesThermal>();
        let mut faces_forces = vec![Quantity::<Power>::default(); self.faces_nodes().len()];
        let mut center_force = Quantity::<Power>::default();
        self.tetrahedra()
            .iter()
            .zip(self.tetrahedra_temperatures(nodal_temperatures))
            .zip(self.tetrahedra_nodes())
            .try_for_each(|((tetrahedron, temperatures), &[face, node_b, node_a])| {
                let nodal_forces = ThermalConductionFiniteElement::nodal_forces(
                    tetrahedron,
                    constitutive_model,
                    &temperatures,
                )?;
                faces_forces[face] += nodal_forces[0];
                forces[node_b] += nodal_forces[1] * stabilization;
                forces[node_a] += nodal_forces[2] * stabilization;
                center_force += nodal_forces[3];
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| self.upstream(error))?;
        self.faces_nodes()
            .iter()
            .zip(faces_forces.iter())
            .for_each(|(face_nodes, &face_force)| {
                let face_force = face_force * (stabilization / face_nodes.len() as Scalar);
                face_nodes
                    .iter()
                    .for_each(|&face_node| forces[face_node] += face_force)
            });
        center_force *= stabilization * inverse_num_nodes;
        forces.iter_mut().for_each(|force| *force += center_force);
        Ok(forces)
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_temperatures: &ElementNodalTemperatures,
    ) -> Result<ElementNodalStiffnessesThermal, VirtualElementError> {
        let num_nodes = nodal_temperatures.len();
        let stabilization = self.stabilization();
        let inverse_num_nodes = 1.0 / num_nodes as Scalar;
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
        let num_faces = self.faces_nodes().len();
        let zero = Quantity::<PowerPerTemperature>::default();
        let mut faces_stiffnesses = vec![zero; num_faces];
        let mut faces_rows = vec![zero; num_faces];
        let mut faces_columns = vec![zero; num_faces];
        let mut rows = vec![zero; num_nodes];
        let mut columns = vec![zero; num_nodes];
        let mut center_stiffness = zero;
        self.tetrahedra()
            .iter()
            .zip(self.tetrahedra_temperatures(nodal_temperatures))
            .zip(self.tetrahedra_nodes())
            .try_for_each(|((tetrahedron, temperatures), &[face, node_b, node_a])| {
                let nodal_stiffnesses = ThermalConductionFiniteElement::nodal_stiffnesses(
                    tetrahedron,
                    constitutive_model,
                    &temperatures,
                )?;
                let face_nodes = &self.faces_nodes()[face];
                let weight = stabilization / face_nodes.len() as Scalar;
                faces_stiffnesses[face] += nodal_stiffnesses[0][0];
                faces_rows[face] += nodal_stiffnesses[0][3];
                faces_columns[face] += nodal_stiffnesses[3][0];
                let face_node_b = nodal_stiffnesses[0][1] * weight;
                let face_node_a = nodal_stiffnesses[0][2] * weight;
                let node_b_face = nodal_stiffnesses[1][0] * weight;
                let node_a_face = nodal_stiffnesses[2][0] * weight;
                face_nodes.iter().for_each(|&face_node| {
                    stiffnesses[face_node][node_b] += face_node_b;
                    stiffnesses[face_node][node_a] += face_node_a;
                    stiffnesses[node_b][face_node] += node_b_face;
                    stiffnesses[node_a][face_node] += node_a_face;
                });
                stiffnesses[node_b][node_b] += nodal_stiffnesses[1][1] * stabilization;
                stiffnesses[node_b][node_a] += nodal_stiffnesses[1][2] * stabilization;
                stiffnesses[node_a][node_b] += nodal_stiffnesses[2][1] * stabilization;
                stiffnesses[node_a][node_a] += nodal_stiffnesses[2][2] * stabilization;
                rows[node_b] += nodal_stiffnesses[1][3] * (stabilization * inverse_num_nodes);
                rows[node_a] += nodal_stiffnesses[2][3] * (stabilization * inverse_num_nodes);
                columns[node_b] += nodal_stiffnesses[3][1] * (stabilization * inverse_num_nodes);
                columns[node_a] += nodal_stiffnesses[3][2] * (stabilization * inverse_num_nodes);
                center_stiffness += nodal_stiffnesses[3][3]
                    * (stabilization * inverse_num_nodes * inverse_num_nodes);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| self.upstream(error))?;
        self.faces_nodes()
            .iter()
            .zip(
                faces_stiffnesses
                    .iter()
                    .zip(faces_rows.iter().zip(faces_columns.iter())),
            )
            .for_each(
                |(face_nodes, (&face_stiffness, (&face_row, &face_column)))| {
                    let inverse_num_nodes_face = 1.0 / face_nodes.len() as Scalar;
                    let face_stiffness = face_stiffness
                        * (stabilization * inverse_num_nodes_face * inverse_num_nodes_face);
                    let face_row =
                        face_row * (stabilization * inverse_num_nodes_face * inverse_num_nodes);
                    let face_column =
                        face_column * (stabilization * inverse_num_nodes_face * inverse_num_nodes);
                    face_nodes.iter().for_each(|&face_node_a| {
                        rows[face_node_a] += face_row;
                        columns[face_node_a] += face_column;
                        face_nodes.iter().for_each(|&face_node_b| {
                            stiffnesses[face_node_a][face_node_b] += face_stiffness
                        })
                    })
                },
            );
        rows.iter_mut().for_each(|row| *row += center_stiffness);
        stiffnesses
            .iter_mut()
            .zip(rows.iter())
            .for_each(|(stiffness, &row)| {
                stiffness
                    .iter_mut()
                    .zip(columns.iter())
                    .for_each(|(entry, &column)| {
                        *entry += row;
                        *entry += column
                    })
            });
        Ok(stiffnesses)
    }
}
