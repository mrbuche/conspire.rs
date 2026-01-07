use crate::{
    constitutive::solid::elastic::Elastic,
    fem::block::element::{FiniteElementError, solid::elastic::ElasticFiniteElement},
    math::{
        ContractSecondFourthIndicesWithFirstIndicesOf, Scalar, Tensor, TensorArray, TensorRank2,
        TensorRank2Vec,
    },
    mechanics::{FirstPiolaKirchhoffStresses, FirstPiolaKirchhoffTangentStiffnesses},
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
        let mut tetrahedra_forces =
            ElementNodalForcesSolid::from(vec![[0.0; 3]; nodal_coordinates.len()]);
        let num_nodes = nodal_coordinates.len() as Scalar;
        match self
            .tetrahedra()
            .iter()
            .zip(self.tetrahedra_coordinates(&nodal_coordinates).iter())
            .zip(self.tetrahedra_nodes.iter())
            .try_for_each(
                |((tetrahedron, tetrahedron_coordinates), &[face, node_b, node_a])| {
                    let num_nodes_face = self.faces_nodes()[face].len() as Scalar;
                    let nodal_forces =
                        tetrahedron.nodal_forces(constitutive_model, tetrahedron_coordinates)?;
                    self.faces_nodes()[face].iter().for_each(|&face_node| {
                        tetrahedra_forces[face_node] += &nodal_forces[0] / num_nodes_face;
                    });
                    tetrahedra_forces[node_b] += &nodal_forces[1];
                    tetrahedra_forces[node_a] += &nodal_forces[2];
                    tetrahedra_forces.iter_mut().for_each(|entry| {
                        *entry += &nodal_forces[3] / num_nodes;
                    });
                    Ok::<(), FiniteElementError>(())
                },
            ) {
            Ok(()) => {
                match self
                    .deformation_gradients(nodal_coordinates)
                    .iter()
                    .map(|deformation_gradient| {
                        constitutive_model.first_piola_kirchhoff_stress(deformation_gradient)
                    })
                    .collect::<Result<FirstPiolaKirchhoffStresses, _>>()
                {
                    Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
                        .iter()
                        .zip(
                            self.gradient_vectors()
                                .iter()
                                .zip(self.integration_weights()),
                        )
                        .map(
                            |(
                                first_piola_kirchhoff_stress,
                                (gradient_vectors, integration_weight),
                            )| {
                                gradient_vectors
                                    .iter()
                                    .map(|gradient_vector| {
                                        (first_piola_kirchhoff_stress * gradient_vector)
                                            * integration_weight
                                    })
                                    .collect()
                            },
                        )
                        .sum::<ElementNodalForcesSolid>()
                        * (1.0 - self.stabilization())
                        + tetrahedra_forces * self.stabilization()),
                    Err(error) => Err(VirtualElementError::Upstream(
                        format!("{error}"),
                        format!("{self:?}"),
                    )),
                }
            }
            Err(error) => Err(VirtualElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses<'a>(
        &'a self,
        constitutive_model: &'a C,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Result<ElementNodalStiffnessesSolid, VirtualElementError> {
        let mut tetrahedra_stiffnesses = ElementNodalStiffnessesSolid::from(vec![
                TensorRank2Vec::from(vec![
                    TensorRank2::zero(); nodal_coordinates.len()
                ]);
                nodal_coordinates.len()
            ]);
        let num_nodes = nodal_coordinates.len() as Scalar;
        match self
            .tetrahedra()
            .iter()
            .zip(self.tetrahedra_coordinates(&nodal_coordinates).iter())
            .zip(self.tetrahedra_nodes.iter())
            .try_for_each(
                |((tetrahedron, tetrahedron_coordinates), &[face, node_b, node_a])| {
                    let num_nodes_face = self.faces_nodes()[face].len() as Scalar;
                    let nodal_stiffnesses = tetrahedron
                        .nodal_stiffnesses(constitutive_model, tetrahedron_coordinates)?;
                    self.faces_nodes()[face].iter().for_each(|&face_node_1| {
                        tetrahedra_stiffnesses[face_node_1][node_b] +=
                            &nodal_stiffnesses[0][1] / num_nodes_face;
                        tetrahedra_stiffnesses[face_node_1][node_a] +=
                            &nodal_stiffnesses[0][2] / num_nodes_face;
                        tetrahedra_stiffnesses[node_b][face_node_1] +=
                            &nodal_stiffnesses[1][0] / num_nodes_face;
                        tetrahedra_stiffnesses[node_a][face_node_1] +=
                            &nodal_stiffnesses[2][0] / num_nodes_face;
                        tetrahedra_stiffnesses[face_node_1]
                            .iter_mut()
                            .for_each(|entry| {
                                *entry += &nodal_stiffnesses[0][3] / num_nodes_face / num_nodes;
                            });
                        self.faces_nodes()[face].iter().for_each(|&face_node_2| {
                            tetrahedra_stiffnesses[face_node_1][face_node_2] +=
                                &nodal_stiffnesses[0][0] / num_nodes_face.powi(2);
                        })
                    });
                    tetrahedra_stiffnesses[node_b][node_b] += &nodal_stiffnesses[1][1];
                    tetrahedra_stiffnesses[node_b][node_a] += &nodal_stiffnesses[1][2];
                    tetrahedra_stiffnesses[node_a][node_b] += &nodal_stiffnesses[2][1];
                    tetrahedra_stiffnesses[node_a][node_a] += &nodal_stiffnesses[2][2];
                    tetrahedra_stiffnesses
                        .iter_mut()
                        .for_each(|tetrahedra_stiffness| {
                            tetrahedra_stiffness[node_b] += &nodal_stiffnesses[3][1] / num_nodes;
                            tetrahedra_stiffness[node_a] += &nodal_stiffnesses[3][2] / num_nodes;
                            self.faces_nodes()[face].iter().for_each(|&face_node| {
                                tetrahedra_stiffness[face_node] +=
                                    &nodal_stiffnesses[3][0] / num_nodes_face / num_nodes;
                            });
                            tetrahedra_stiffness.iter_mut().for_each(|entry| {
                                *entry += &nodal_stiffnesses[3][3] / num_nodes.powi(2);
                            })
                        });
                    tetrahedra_stiffnesses[node_b].iter_mut().for_each(|entry| {
                        *entry += &nodal_stiffnesses[1][3] / num_nodes;
                    });
                    tetrahedra_stiffnesses[node_a].iter_mut().for_each(|entry| {
                        *entry += &nodal_stiffnesses[2][3] / num_nodes;
                    });
                    Ok::<(), FiniteElementError>(())
                },
            ) {
            Ok(()) => {
                match self
                    .deformation_gradients(nodal_coordinates)
                    .iter()
                    .map(|deformation_gradient| {
                        constitutive_model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)
                    })
                    .collect::<Result<FirstPiolaKirchhoffTangentStiffnesses, _>>()
                {
                    Ok(first_piola_kirchhoff_tangent_stiffnesses) => {
                        Ok(first_piola_kirchhoff_tangent_stiffnesses
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
                                    gradient_vectors
                                        .iter()
                                        .map(|gradient_vector_a| {
                                            gradient_vectors
                                                .iter()
                                                .map(|gradient_vector_b| {
                                                    first_piola_kirchhoff_tangent_stiffness
                                                    .contract_second_fourth_indices_with_first_indices_of(
                                                        gradient_vector_a,
                                                        gradient_vector_b,
                                                    )
                                                    * integration_weight
                                                })
                                                .collect()
                                        })
                                        .collect()
                                },
                            )
                            .sum::<ElementNodalStiffnessesSolid>()
                            * (1.0 - self.stabilization())
                            + tetrahedra_stiffnesses * self.stabilization())
                    }
                    Err(error) => Err(VirtualElementError::Upstream(
                        format!("{error}"),
                        format!("{self:?}"),
                    )),
                }
            }
            Err(error) => Err(VirtualElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
