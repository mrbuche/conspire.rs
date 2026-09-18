use super::{super::Node, SolidElement};
use crate::{
    constitutive::{ConstitutiveError, solid::elastic::Elastic},
    domain::NodalCoordinates,
    math::{ContractSecondFourthWithFirst, Current, TensorRank1, TensorRank2},
    units::{Force, ForcePerLength},
};

pub use crate::domain::block::element::solid::elastic::ElasticElement;

/// A nodal force contribution, ordered the same as [`Node::gradient_vectors`].
type NodalForce = TensorRank1<3, Current, Force>;
/// A nodal stiffness block, ordered the same as [`Node::gradient_vectors`]
/// on both axes.
type NodalStiffness = TensorRank2<3, Current, Current, ForcePerLength>;

impl<C> ElasticElement<C> for Node
where
    C: Elastic,
{
    type Forces = Vec<NodalForce>;
    type Stiffnesses = Vec<Vec<NodalStiffness>>;
    type Error = ConstitutiveError;
    /// The forces this particle's stress contributes to each of its bonded
    /// neighbors (including itself), ordered the same as its bond list.
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
            .map(|(_, bond_gradient_vector)| {
                (&first_piola_kirchhoff_stress * bond_gradient_vector) * self.volume
            })
            .collect())
    }
    /// The stiffness blocks this particle's tangent contributes between each
    /// pair of its bonded neighbors (including itself), ordered the same as
    /// its bond list on both axes.
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
            .map(|(_, bond_gradient_vector_a)| {
                self.gradient_vectors()
                    .iter()
                    .map(|(_, bond_gradient_vector_b)| {
                        first_piola_kirchhoff_tangent_stiffness.contract_second_fourth_with_first(
                            bond_gradient_vector_a,
                            bond_gradient_vector_b,
                        ) * self.volume
                    })
                    .collect()
            })
            .collect())
    }
}
