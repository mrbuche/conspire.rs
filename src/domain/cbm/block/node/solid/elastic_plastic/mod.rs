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
    fn nodal_forces_and_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariables<1>,
    ) -> Result<(Vec<NodalForce>, Vec<Vec<NodalStiffness>>), ConstitutiveError> {
        let (first_piola_kirchhoff_stress, first_piola_kirchhoff_tangent_stiffness, _) =
            constitutive_model.condensed(
                &self.deformation_gradients(nodal_coordinates),
                &state_variables[0],
            )?;
        Ok((
            self.gradient_vectors()
                .iter()
                .map(|gradient_vector| {
                    (&first_piola_kirchhoff_stress * gradient_vector) * self.volume
                })
                .collect(),
            self.gradient_vectors()
                .iter()
                .map(|gradient_vector_a| {
                    self.gradient_vectors()
                        .iter()
                        .map(|gradient_vector_b| {
                            first_piola_kirchhoff_tangent_stiffness
                                .contract_second_fourth_with_first(
                                    gradient_vector_a,
                                    gradient_vector_b,
                                )
                                * self.volume
                        })
                        .collect()
                })
                .collect(),
        ))
    }
    fn updated_state(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariables<1>,
    ) -> Result<PlasticStateVariables<1>, ConstitutiveError> {
        let (_, _, state) = constitutive_model.condensed(
            &self.deformation_gradients(nodal_coordinates),
            &state_variables[0],
        )?;
        Ok([state].into())
    }
}
