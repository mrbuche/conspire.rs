use super::{super::Node, SolidElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic_plastic::ElasticPlastic},
    domain::{NodalCoordinates, block::element::solid::plastic::PlasticStateVariables},
    math::{ContractSecondFourthWithFirst, Current, TensorRank1, TensorRank2},
    units::{Force, ForcePerLength},
};

pub use crate::domain::block::element::solid::elastic_plastic::ElasticPlasticElement;

type NodalForce = TensorRank1<3, Current, Force>;
type NodalStiffness = TensorRank2<3, Current, Current, ForcePerLength>;

impl<C> ElasticPlasticElement<C, 1> for Node
where
    C: ElasticPlastic,
{
    type Forces = Vec<NodalForce>;
    type Stiffnesses = Vec<Vec<NodalStiffness>>;
    type Error = ConstitutiveError;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariables<1>,
    ) -> Result<Vec<NodalForce>, ConstitutiveError> {
        let deformation_gradient = self.deformation_gradients(nodal_coordinates);
        let updated = constitutive_model.return_map(&deformation_gradient, &state_variables[0])?;
        let first_piola_kirchhoff_stress =
            constitutive_model.first_piola_kirchhoff_stress(&deformation_gradient, &updated.0)?;
        Ok(self
            .gradient_vectors()
            .iter()
            .map(|gradient_vector| (&first_piola_kirchhoff_stress * gradient_vector) * self.volume)
            .collect())
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariables<1>,
    ) -> Result<Vec<Vec<NodalStiffness>>, ConstitutiveError> {
        let (first_piola_kirchhoff_tangent_stiffness, _) = constitutive_model
            .consistent_tangent_stiffness(
                &self.deformation_gradients(nodal_coordinates),
                &state_variables[0],
            )?;
        Ok(self
            .gradient_vectors()
            .iter()
            .map(|gradient_vector_a| {
                self.gradient_vectors()
                    .iter()
                    .map(|gradient_vector_b| {
                        first_piola_kirchhoff_tangent_stiffness
                            .contract_second_fourth_with_first(gradient_vector_a, gradient_vector_b)
                            * self.volume
                    })
                    .collect()
            })
            .collect())
    }
    fn updated_state(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariables<1>,
    ) -> Result<PlasticStateVariables<1>, ConstitutiveError> {
        Ok([constitutive_model.return_map(
            &self.deformation_gradients(nodal_coordinates),
            &state_variables[0],
        )?]
        .into())
    }
}
