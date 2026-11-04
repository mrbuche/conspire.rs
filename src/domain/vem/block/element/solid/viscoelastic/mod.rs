use crate::{
    constitutive::solid::viscoelastic::Viscoelastic,
    domain::block::element::solid::viscoelastic::ViscoelasticElement,
    fem::block::element::FiniteElementError,
    math::{ContractSecondFourthWithFirst, Scalar, Tensor, TensorArray},
    mechanics::{
        Damping, FirstPiolaKirchhoffRateTangentStiffnesses, FirstPiolaKirchhoffStresses,
        FirstPiolaKirchhoffTangentStiffnesses, Force,
    },
    vem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, VirtualElement,
        VirtualElementError,
        solid::{
            ElementNodalDampingsSolid, ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
            SolidElement, SolidVirtualElement,
        },
    },
};

pub trait ViscoelasticVirtualElement<C>
where
    C: Viscoelastic,
    Self: SolidVirtualElement
        + ViscoelasticElement<
            C,
            Forces = ElementNodalForcesSolid,
            Stiffnesses = ElementNodalStiffnessesSolid,
            Dampings = ElementNodalDampingsSolid,
            Error = VirtualElementError,
        >,
{
}

impl<T, C> ViscoelasticVirtualElement<C> for T
where
    C: Viscoelastic,
    T: SolidVirtualElement
        + ViscoelasticElement<
            C,
            Forces = ElementNodalForcesSolid,
            Stiffnesses = ElementNodalStiffnessesSolid,
            Dampings = ElementNodalDampingsSolid,
            Error = VirtualElementError,
        >,
{
}

impl<C> ViscoelasticElement<C> for Element
where
    C: Viscoelastic,
{
    type Forces = ElementNodalForcesSolid;
    type Stiffnesses = ElementNodalStiffnessesSolid;
    type Dampings = ElementNodalDampingsSolid;
    type Error = VirtualElementError;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
        nodal_velocities: &ElementNodalVelocities,
    ) -> Result<ElementNodalForcesSolid, VirtualElementError> {
        let stabilization = self.stabilization();
        let inverse_num_nodes = 1.0 / nodal_coordinates.len() as Scalar;
        let tetrahedra_coordinates = self.tetrahedra_coordinates(nodal_coordinates);
        let tetrahedra_velocities = self.tetrahedra_coordinates(nodal_velocities);
        let mut forces = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter(),
            )
            .map(|(deformation_gradient, deformation_gradient_rate)| {
                constitutive_model
                    .first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_rate)
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
            .zip(
                tetrahedra_coordinates
                    .iter()
                    .zip(tetrahedra_velocities.iter()),
            )
            .zip(self.tetrahedra_nodes().iter())
            .try_for_each(
                |(
                    (tetrahedron, (tetrahedron_coordinates, tetrahedron_velocities)),
                    &[face, node_b, node_a],
                )| {
                    let nodal_forces = tetrahedron.nodal_forces(
                        constitutive_model,
                        tetrahedron_coordinates,
                        tetrahedron_velocities,
                    )?;
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
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
        nodal_velocities: &ElementNodalVelocities,
    ) -> Result<ElementNodalStiffnessesSolid, VirtualElementError> {
        let tetrahedra_velocities = self.tetrahedra_coordinates(nodal_velocities);
        let tangents = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter(),
            )
            .map(|(deformation_gradient, deformation_gradient_rate)| {
                constitutive_model.first_piola_kirchhoff_tangent_stiffness(
                    deformation_gradient,
                    deformation_gradient_rate,
                )
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses, _>>()
            .map_err(|error| self.upstream(error))?;
        let tetrahedra = self
            .tetrahedra()
            .iter()
            .zip(
                self.tetrahedra_coordinates(nodal_coordinates)
                    .iter()
                    .zip(tetrahedra_velocities.iter()),
            )
            .map(
                |(tetrahedron, (tetrahedron_coordinates, tetrahedron_velocities))| {
                    tetrahedron.nodal_stiffnesses(
                        constitutive_model,
                        tetrahedron_coordinates,
                        tetrahedron_velocities,
                    )
                },
            )
            .collect::<Result<Vec<_>, FiniteElementError>>()
            .map_err(|error| self.upstream(error))?;
        self.stabilized_stiffnesses(
            &tangents,
            &tetrahedra.iter().collect::<Vec<_>>(),
            nodal_coordinates.len(),
        )
    }
    fn nodal_dampings(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
        nodal_velocities: &ElementNodalVelocities,
    ) -> Result<ElementNodalDampingsSolid, VirtualElementError> {
        let num_nodes = nodal_coordinates.len();
        let stabilization = self.stabilization();
        let inverse_num_nodes = 1.0 / num_nodes as Scalar;
        let tetrahedra_coordinates = self.tetrahedra_coordinates(nodal_coordinates);
        let tetrahedra_velocities = self.tetrahedra_coordinates(nodal_velocities);
        let mut dampings = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter(),
            )
            .map(|(deformation_gradient, deformation_gradient_rate)| {
                constitutive_model.first_piola_kirchhoff_rate_tangent_stiffness(
                    deformation_gradient,
                    deformation_gradient_rate,
                )
            })
            .collect::<Result<FirstPiolaKirchhoffRateTangentStiffnesses, _>>()
            .map_err(|error| self.upstream(error))?
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(
                    first_piola_kirchhoff_rate_tangent_stiffness,
                    (gradient_vectors, integration_weight),
                )| {
                    let weight = integration_weight * (1.0 - stabilization);
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector_a| {
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_b| {
                                    first_piola_kirchhoff_rate_tangent_stiffness
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
            .sum::<ElementNodalDampingsSolid>();
        let num_faces = self.faces_nodes().len();
        let mut faces_dampings = vec![Damping::zero(); num_faces];
        let mut faces_rows = vec![Damping::zero(); num_faces];
        let mut faces_columns = vec![Damping::zero(); num_faces];
        let mut rows = vec![Damping::zero(); num_nodes];
        let mut columns = vec![Damping::zero(); num_nodes];
        let mut center_damping = Damping::zero();
        self.tetrahedra()
            .iter()
            .zip(
                tetrahedra_coordinates
                    .iter()
                    .zip(tetrahedra_velocities.iter()),
            )
            .zip(self.tetrahedra_nodes().iter())
            .try_for_each(
                |(
                    (tetrahedron, (tetrahedron_coordinates, tetrahedron_velocities)),
                    &[face, node_b, node_a],
                )| {
                    let tetrahedron_dampings = tetrahedron.nodal_dampings(
                        constitutive_model,
                        tetrahedron_coordinates,
                        tetrahedron_velocities,
                    )?;
                    let face_nodes = &self.faces_nodes()[face];
                    let weight = stabilization / face_nodes.len() as Scalar;
                    faces_dampings[face] += &tetrahedron_dampings[0][0];
                    faces_rows[face] += &tetrahedron_dampings[0][3];
                    faces_columns[face] += &tetrahedron_dampings[3][0];
                    let face_node_b = &tetrahedron_dampings[0][1] * weight;
                    let face_node_a = &tetrahedron_dampings[0][2] * weight;
                    let node_b_face = &tetrahedron_dampings[1][0] * weight;
                    let node_a_face = &tetrahedron_dampings[2][0] * weight;
                    face_nodes.iter().for_each(|&face_node| {
                        dampings[face_node][node_b] += &face_node_b;
                        dampings[face_node][node_a] += &face_node_a;
                        dampings[node_b][face_node] += &node_b_face;
                        dampings[node_a][face_node] += &node_a_face;
                    });
                    dampings[node_b][node_b] += &tetrahedron_dampings[1][1] * stabilization;
                    dampings[node_b][node_a] += &tetrahedron_dampings[1][2] * stabilization;
                    dampings[node_a][node_b] += &tetrahedron_dampings[2][1] * stabilization;
                    dampings[node_a][node_a] += &tetrahedron_dampings[2][2] * stabilization;
                    rows[node_b] +=
                        &tetrahedron_dampings[1][3] * (stabilization * inverse_num_nodes);
                    rows[node_a] +=
                        &tetrahedron_dampings[2][3] * (stabilization * inverse_num_nodes);
                    columns[node_b] +=
                        &tetrahedron_dampings[3][1] * (stabilization * inverse_num_nodes);
                    columns[node_a] +=
                        &tetrahedron_dampings[3][2] * (stabilization * inverse_num_nodes);
                    center_damping += &tetrahedron_dampings[3][3]
                        * (stabilization * inverse_num_nodes * inverse_num_nodes);
                    Ok::<(), FiniteElementError>(())
                },
            )
            .map_err(|error| self.upstream(error))?;
        self.faces_nodes()
            .iter()
            .zip(
                faces_dampings
                    .iter()
                    .zip(faces_rows.iter().zip(faces_columns.iter())),
            )
            .for_each(|(face_nodes, (face_damping, (face_row, face_column)))| {
                let inverse_num_nodes_face = 1.0 / face_nodes.len() as Scalar;
                let face_damping = face_damping
                    * (stabilization * inverse_num_nodes_face * inverse_num_nodes_face);
                let face_row =
                    face_row * (stabilization * inverse_num_nodes_face * inverse_num_nodes);
                let face_column =
                    face_column * (stabilization * inverse_num_nodes_face * inverse_num_nodes);
                face_nodes.iter().for_each(|&face_node_a| {
                    rows[face_node_a] += &face_row;
                    columns[face_node_a] += &face_column;
                    face_nodes.iter().for_each(|&face_node_b| {
                        dampings[face_node_a][face_node_b] += &face_damping
                    })
                })
            });
        rows.iter_mut().for_each(|row| *row += &center_damping);
        dampings
            .iter_mut()
            .zip(rows.iter())
            .for_each(|(damping, row)| {
                damping
                    .iter_mut()
                    .zip(columns.iter())
                    .for_each(|(entry, column)| {
                        *entry += row;
                        *entry += column
                    })
            });
        Ok(dampings)
    }
}
