use super::{super::Node, SolidElement};
use crate::{
    constitutive::{ConstitutiveError, solid::viscoelastic::Viscoelastic},
    domain::{NodalCoordinates, NodalVelocities},
    math::{ContractSecondFourthWithFirst, Current, TensorRank1, TensorRank2},
    units::{Force, ForcePerVelocity},
};

pub use crate::domain::block::element::solid::viscoelastic::ViscoelasticElement;

type NodalForce = TensorRank1<3, Current, Force>;
type NodalDamping = TensorRank2<3, Current, Current, ForcePerVelocity>;

impl<C> ViscoelasticElement<C> for Node
where
    C: Viscoelastic,
{
    type Forces = Vec<NodalForce>;
    type Dampings = Vec<Vec<NodalDamping>>;
    type Error = ConstitutiveError;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Vec<NodalForce>, ConstitutiveError> {
        let first_piola_kirchhoff_stress = constitutive_model.first_piola_kirchhoff_stress(
            &self.deformation_gradients(nodal_coordinates),
            &self.deformation_gradient_rates(nodal_coordinates, nodal_velocities),
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
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Vec<Vec<NodalDamping>>, ConstitutiveError> {
        let first_piola_kirchhoff_rate_tangent_stiffness = constitutive_model
            .first_piola_kirchhoff_rate_tangent_stiffness(
                &self.deformation_gradients(nodal_coordinates),
                &self.deformation_gradient_rates(nodal_coordinates, nodal_velocities),
            )?;
        Ok(self
            .gradient_vectors()
            .iter()
            .map(|gradient_vector_a| {
                self.gradient_vectors()
                    .iter()
                    .map(|gradient_vector_b| {
                        first_piola_kirchhoff_rate_tangent_stiffness
                            .contract_second_fourth_with_first(gradient_vector_a, gradient_vector_b)
                            * self.volume
                    })
                    .collect()
            })
            .collect())
    }
}
