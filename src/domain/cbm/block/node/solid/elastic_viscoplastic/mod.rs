use super::{super::Node, SolidElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic_viscoplastic::ElasticViscoplastic},
    domain::{
        NodalCoordinates,
        block::element::solid::viscoplastic::{ViscoplasticEvolution, ViscoplasticStateVariables},
    },
    math::{
        ContractSecondFourthWithFirst, Current, Differentiable, Tensor, TensorRank1, TensorRank2,
    },
    units::{Force, ForcePerLength},
};

pub use crate::domain::block::element::solid::elastic_viscoplastic::ElasticViscoplasticElement;

type NodalForce = TensorRank1<3, Current, Force>;
type NodalStiffness = TensorRank2<3, Current, Current, ForcePerLength>;

impl<C, Y> ElasticViscoplasticElement<C, 1, Y> for Node
where
    C: ElasticViscoplastic<Y>,
    Y: Differentiable + Tensor,
{
    type Forces = Vec<NodalForce>;
    type Stiffnesses = Vec<Vec<NodalStiffness>>;
    type Error = ConstitutiveError;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<1, Y>,
    ) -> Result<Vec<NodalForce>, ConstitutiveError> {
        let (deformation_gradient_p, _) = (&state_variables[0]).into();
        let first_piola_kirchhoff_stress = constitutive_model.first_piola_kirchhoff_stress(
            &self.deformation_gradients(nodal_coordinates),
            deformation_gradient_p,
        )?;
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
        state_variables: &ViscoplasticStateVariables<1, Y>,
    ) -> Result<Vec<Vec<NodalStiffness>>, ConstitutiveError> {
        let (deformation_gradient_p, _) = (&state_variables[0]).into();
        let first_piola_kirchhoff_tangent_stiffness = constitutive_model
            .first_piola_kirchhoff_tangent_stiffness(
                &self.deformation_gradients(nodal_coordinates),
                deformation_gradient_p,
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
    fn state_variables_evolution(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<1, Y>,
    ) -> Result<ViscoplasticEvolution<1, Y>, ConstitutiveError> {
        Ok([constitutive_model.state_variables_evolution(
            &self.deformation_gradients(nodal_coordinates),
            &state_variables[0],
        )?]
        .into())
    }
}
