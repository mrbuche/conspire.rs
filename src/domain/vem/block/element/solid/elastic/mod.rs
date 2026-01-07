use crate::{
    constitutive::solid::elastic::Elastic,
    fem::block::element::{FiniteElementError, solid::elastic::ElasticFiniteElement},
    math::{ContractSecondFourthIndicesWithFirstIndicesOf, Scalar, Tensor},
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
        //
        // GET RID OF UNWRAPS
        //
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
                    .sum())
            }
            Err(error) => Err(VirtualElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
