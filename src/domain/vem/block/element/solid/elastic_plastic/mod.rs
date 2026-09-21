use crate::{
    constitutive::solid::elastic_plastic::ElasticPlastic,
    domain::block::element::solid::{
        elastic_plastic::ElasticPlasticElement, plastic::PlasticStateVariables,
    },
    fem::block::element::{
        FiniteElementError,
        solid::{
            ElementNodalForcesSolid as TetrahedronForces,
            ElementNodalStiffnessesSolid as TetrahedronStiffnesses,
        },
    },
    math::{ContractSecondFourthWithFirst, Scalar, Tensor, TensorArray},
    mechanics::{
        FirstPiolaKirchhoffStresses, FirstPiolaKirchhoffTangentStiffnesses, Force, Stiffness,
    },
    vem::block::element::{
        Element, ElementNodalCoordinates, VirtualElement, VirtualElementError,
        solid::{
            ElementNodalForcesSolid, ElementNodalStiffnessesSolid, SolidElement,
            SolidVirtualElement,
        },
    },
};

pub trait ElasticPlasticVirtualElement<C>
where
    C: ElasticPlastic,
    Self: SolidVirtualElement
        + ElasticPlasticElement<
            C,
            1,
            Forces = ElementNodalForcesSolid,
            Stiffnesses = ElementNodalStiffnessesSolid,
            Error = VirtualElementError,
        >,
{
}

impl<T, C> ElasticPlasticVirtualElement<C> for T
where
    C: ElasticPlastic,
    T: SolidVirtualElement
        + ElasticPlasticElement<
            C,
            1,
            Forces = ElementNodalForcesSolid,
            Stiffnesses = ElementNodalStiffnessesSolid,
            Error = VirtualElementError,
        >,
{
}

impl Element {
    fn stabilized_forces(
        &self,
        stresses: &FirstPiolaKirchhoffStresses,
        tetrahedra_forces: &[&TetrahedronForces<4>],
        num_nodes: usize,
    ) -> Result<ElementNodalForcesSolid, VirtualElementError> {
        let stabilization = self.stabilization();
        let inverse_num_nodes = 1.0 / num_nodes as Scalar;
        let mut forces = stresses
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(first_piola_kirchhoff_stress, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector| {
                            (first_piola_kirchhoff_stress * gradient_vector)
                                * (integration_weight * (1.0 - stabilization))
                        })
                        .collect()
                },
            )
            .sum::<ElementNodalForcesSolid>();
        let mut faces_forces = vec![Force::zero(); self.faces_nodes().len()];
        let mut center_force = Force::zero();
        self.tetrahedra_nodes()
            .iter()
            .zip(tetrahedra_forces.iter())
            .try_for_each(|(&[face, node_b, node_a], nodal_forces)| {
                faces_forces[face] += &nodal_forces[0];
                forces[node_b] += &nodal_forces[1] * stabilization;
                forces[node_a] += &nodal_forces[2] * stabilization;
                center_force += &nodal_forces[3];
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| self.upstream(error))?;
        self.faces_nodes()
            .iter()
            .zip(faces_forces.iter())
            .for_each(|(face_nodes, face_force)| {
                let face_force = face_force * (stabilization / face_nodes.len() as Scalar);
                face_nodes
                    .iter()
                    .for_each(|&face_node| forces[face_node] += &face_force)
            });
        center_force *= stabilization * inverse_num_nodes;
        forces.iter_mut().for_each(|force| *force += &center_force);
        Ok(forces)
    }
    fn stabilized_stiffnesses(
        &self,
        tangents: &FirstPiolaKirchhoffTangentStiffnesses,
        tetrahedra_stiffnesses: &[&TetrahedronStiffnesses<4>],
        num_nodes: usize,
    ) -> Result<ElementNodalStiffnessesSolid, VirtualElementError> {
        let stabilization = self.stabilization();
        let inverse_num_nodes = 1.0 / num_nodes as Scalar;
        let mut stiffnesses = tangents
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(
                    first_piola_kirchhoff_tangent_stiffness,
                    (gradient_vectors, integration_weight),
                )| {
                    let weight = integration_weight * (1.0 - stabilization);
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector_a| {
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_b| {
                                    first_piola_kirchhoff_tangent_stiffness
                                        .contract_second_fourth_with_first(
                                            gradient_vector_a,
                                            gradient_vector_b,
                                        )
                                        * weight
                                })
                                .collect()
                        })
                        .collect()
                },
            )
            .sum::<ElementNodalStiffnessesSolid>();
        let num_faces = self.faces_nodes().len();
        let mut faces_stiffnesses = vec![Stiffness::zero(); num_faces];
        let mut faces_rows = vec![Stiffness::zero(); num_faces];
        let mut faces_columns = vec![Stiffness::zero(); num_faces];
        let mut rows = vec![Stiffness::zero(); num_nodes];
        let mut columns = vec![Stiffness::zero(); num_nodes];
        let mut center_stiffness = Stiffness::zero();
        self.tetrahedra_nodes()
            .iter()
            .zip(tetrahedra_stiffnesses.iter())
            .try_for_each(|(&[face, node_b, node_a], nodal_stiffnesses)| {
                let face_nodes = &self.faces_nodes()[face];
                let weight = stabilization / face_nodes.len() as Scalar;
                faces_stiffnesses[face] += &nodal_stiffnesses[0][0];
                faces_rows[face] += &nodal_stiffnesses[0][3];
                faces_columns[face] += &nodal_stiffnesses[3][0];
                let face_node_b = &nodal_stiffnesses[0][1] * weight;
                let face_node_a = &nodal_stiffnesses[0][2] * weight;
                let node_b_face = &nodal_stiffnesses[1][0] * weight;
                let node_a_face = &nodal_stiffnesses[2][0] * weight;
                face_nodes.iter().for_each(|&face_node| {
                    stiffnesses[face_node][node_b] += &face_node_b;
                    stiffnesses[face_node][node_a] += &face_node_a;
                    stiffnesses[node_b][face_node] += &node_b_face;
                    stiffnesses[node_a][face_node] += &node_a_face;
                });
                stiffnesses[node_b][node_b] += &nodal_stiffnesses[1][1] * stabilization;
                stiffnesses[node_b][node_a] += &nodal_stiffnesses[1][2] * stabilization;
                stiffnesses[node_a][node_b] += &nodal_stiffnesses[2][1] * stabilization;
                stiffnesses[node_a][node_a] += &nodal_stiffnesses[2][2] * stabilization;
                rows[node_b] += &nodal_stiffnesses[1][3] * (stabilization * inverse_num_nodes);
                rows[node_a] += &nodal_stiffnesses[2][3] * (stabilization * inverse_num_nodes);
                columns[node_b] += &nodal_stiffnesses[3][1] * (stabilization * inverse_num_nodes);
                columns[node_a] += &nodal_stiffnesses[3][2] * (stabilization * inverse_num_nodes);
                center_stiffness += &nodal_stiffnesses[3][3]
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
            .for_each(|(face_nodes, (face_stiffness, (face_row, face_column)))| {
                let inverse_num_nodes_face = 1.0 / face_nodes.len() as Scalar;
                let face_stiffness = face_stiffness
                    * (stabilization * inverse_num_nodes_face * inverse_num_nodes_face);
                let face_row =
                    face_row * (stabilization * inverse_num_nodes_face * inverse_num_nodes);
                let face_column =
                    face_column * (stabilization * inverse_num_nodes_face * inverse_num_nodes);
                face_nodes.iter().for_each(|&face_node_a| {
                    rows[face_node_a] += &face_row;
                    columns[face_node_a] += &face_column;
                    face_nodes.iter().for_each(|&face_node_b| {
                        stiffnesses[face_node_a][face_node_b] += &face_stiffness
                    })
                })
            });
        rows.iter_mut().for_each(|row| *row += &center_stiffness);
        stiffnesses
            .iter_mut()
            .zip(rows.iter())
            .for_each(|(stiffness, row)| {
                stiffness
                    .iter_mut()
                    .zip(columns.iter())
                    .for_each(|(entry, column)| {
                        *entry += row;
                        *entry += column
                    })
            });
        Ok(stiffnesses)
    }
}

impl<C> ElasticPlasticElement<C, 1> for Element
where
    C: ElasticPlastic,
{
    type Forces = ElementNodalForcesSolid;
    type Stiffnesses = ElementNodalStiffnessesSolid;
    type Error = VirtualElementError;
    fn nodal_forces_and_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
        state_variables: &PlasticStateVariables<1>,
    ) -> Result<(ElementNodalForcesSolid, ElementNodalStiffnessesSolid), VirtualElementError> {
        let evaluations = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .map(|(deformation_gradient, state_variable)| {
                constitutive_model
                    .condensed(deformation_gradient, state_variable)
                    .map(|(stress, tangent, _)| (stress, tangent))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| self.upstream(error))?;
        let stresses = evaluations
            .iter()
            .map(|(stress, _)| stress.clone())
            .collect::<FirstPiolaKirchhoffStresses>();
        let tangents = evaluations
            .into_iter()
            .map(|(_, tangent)| tangent)
            .collect::<FirstPiolaKirchhoffTangentStiffnesses>();
        let tetrahedra = self
            .tetrahedra()
            .iter()
            .zip(self.tetrahedra_coordinates(nodal_coordinates).iter())
            .map(|(tetrahedron, tetrahedron_coordinates)| {
                tetrahedron.nodal_forces_and_stiffnesses(
                    constitutive_model,
                    tetrahedron_coordinates,
                    state_variables,
                )
            })
            .collect::<Result<Vec<_>, FiniteElementError>>()
            .map_err(|error| self.upstream(error))?;
        let num_nodes = nodal_coordinates.len();
        Ok((
            self.stabilized_forces(
                &stresses,
                &tetrahedra
                    .iter()
                    .map(|(forces, _)| forces)
                    .collect::<Vec<_>>(),
                num_nodes,
            )?,
            self.stabilized_stiffnesses(
                &tangents,
                &tetrahedra
                    .iter()
                    .map(|(_, stiffnesses)| stiffnesses)
                    .collect::<Vec<_>>(),
                num_nodes,
            )?,
        ))
    }
    fn updated_state(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
        state_variables: &PlasticStateVariables<1>,
    ) -> Result<PlasticStateVariables<1>, VirtualElementError> {
        self.deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .map(|(deformation_gradient, state_variable)| {
                constitutive_model
                    .condensed(deformation_gradient, state_variable)
                    .map(|(_, _, state)| state)
            })
            .collect::<Result<PlasticStateVariables<1>, _>>()
            .map_err(|error| self.upstream(error))
    }
}
