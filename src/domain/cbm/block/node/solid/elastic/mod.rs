use super::{super::Node, SolidElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic::Elastic},
    domain::NodalCoordinates,
    math::{ContractSecondFourthWithFirst, Current, TensorRank1, TensorRank2},
    units::{Force, ForcePerLength},
};

pub use crate::domain::block::element::solid::elastic::ElasticElement;

type NodalForce = TensorRank1<3, Current, Force>;
type NodalStiffness = TensorRank2<3, Current, Current, ForcePerLength>;

impl<C> ElasticElement<C> for Node
where
    C: Elastic,
{
    type Forces = Vec<NodalForce>;
    type Stiffnesses = Vec<Vec<NodalStiffness>>;
    type Error = ConstitutiveError;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Vec<NodalForce>, ConstitutiveError> {
        let first_piola_kirchhoff_stress = constitutive_model
            .first_piola_kirchhoff_stress(&self.deformation_gradients(nodal_coordinates))?;
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
    ) -> Result<Vec<Vec<NodalStiffness>>, ConstitutiveError> {
        let first_piola_kirchhoff_tangent_stiffness = constitutive_model
            .first_piola_kirchhoff_tangent_stiffness(
                &self.deformation_gradients(nodal_coordinates),
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
}
