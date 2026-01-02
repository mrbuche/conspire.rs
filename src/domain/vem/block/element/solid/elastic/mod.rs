use crate::{
    constitutive::solid::elastic::Elastic,
    math::{ContractSecondFourthIndicesWithFirstIndicesOf, Tensor},
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
        // need stabilization terms
        //
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
                    |(first_piola_kirchhoff_stress, (gradient_vectors, integration_weight))| {
                        gradient_vectors
                            .iter()
                            .map(|gradient_vector| {
                                (first_piola_kirchhoff_stress * gradient_vector)
                                    * integration_weight
                            })
                            .collect()
                    },
                )
                .sum()),
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
        //
        // need stabilization terms
        //
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
