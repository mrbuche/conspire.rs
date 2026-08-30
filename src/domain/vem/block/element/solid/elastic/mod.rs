use crate::{
    constitutive::solid::elastic::Elastic,
    fem::block::element::{FiniteElementError, solid::elastic::ElasticFiniteElement},
    math::{ContractSecondFourthWithFirst, Scalar, Tensor, TensorArray},
    mechanics::{
        FirstPiolaKirchhoffStresses, FirstPiolaKirchhoffTangentStiffnesses, Force, Stiffness,
    },
    vem::block::element::{
        Element, ElementNodalCoordinates, VirtualElement, VirtualElementError,
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid, SolidVirtualElement},
    },
};

pub trait ElasticVirtualElement<C>
where
    C: Elastic,
    Self: SolidVirtualElement,
{
    fn nodal_forces<'a>(
        &'a self,
        constitutive_model: &'a C,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Result<ElementNodalForcesSolid, VirtualElementError>;
    fn nodal_stiffnesses<'a>(
        &'a self,
        constitutive_model: &'a C,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Result<ElementNodalStiffnessesSolid, VirtualElementError>;
}

impl<C> ElasticVirtualElement<C> for Element
where
    C: Elastic,
{
    fn nodal_forces<'a>(
        &'a self,
        constitutive_model: &'a C,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Result<ElementNodalForcesSolid, VirtualElementError> {
        let stabilization = self.stabilization();
        let inverse_num_nodes = 1.0 / nodal_coordinates.len() as Scalar;
        let tetrahedra_coordinates = self.tetrahedra_coordinates(&nodal_coordinates);
        let mut forces = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_stress(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffStresses, _>>()
            .map_err(|error| self.upstream(error))?
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
        self.tetrahedra()
            .iter()
            .zip(tetrahedra_coordinates.iter())
            .zip(self.tetrahedra_nodes().iter())
            .try_for_each(
                |((tetrahedron, tetrahedron_coordinates), &[face, node_b, node_a])| {
                    let nodal_forces =
                        tetrahedron.nodal_forces(constitutive_model, tetrahedron_coordinates)?;
                    faces_forces[face] += &nodal_forces[0];
                    forces[node_b] += &nodal_forces[1] * stabilization;
                    forces[node_a] += &nodal_forces[2] * stabilization;
                    center_force += &nodal_forces[3];
                    Ok::<(), FiniteElementError>(())
                },
            )
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
    fn nodal_stiffnesses<'a>(
        &'a self,
        constitutive_model: &'a C,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Result<ElementNodalStiffnessesSolid, VirtualElementError> {
        let num_nodes = nodal_coordinates.len();
        let stabilization = self.stabilization();
        let inverse_num_nodes = 1.0 / num_nodes as Scalar;
        let tetrahedra_coordinates = self.tetrahedra_coordinates(&nodal_coordinates);
        let mut stiffnesses = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .map(|deformation_gradient| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses, _>>()
            .map_err(|error| self.upstream(error))?
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
        self.tetrahedra()
            .iter()
            .zip(tetrahedra_coordinates.iter())
            .zip(self.tetrahedra_nodes().iter())
            .try_for_each(
                |((tetrahedron, tetrahedron_coordinates), &[face, node_b, node_a])| {
                    let nodal_stiffnesses = tetrahedron
                        .nodal_stiffnesses(constitutive_model, tetrahedron_coordinates)?;
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
                    columns[node_b] +=
                        &nodal_stiffnesses[3][1] * (stabilization * inverse_num_nodes);
                    columns[node_a] +=
                        &nodal_stiffnesses[3][2] * (stabilization * inverse_num_nodes);
                    center_stiffness += &nodal_stiffnesses[3][3]
                        * (stabilization * inverse_num_nodes * inverse_num_nodes);
                    Ok::<(), FiniteElementError>(())
                },
            )
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
