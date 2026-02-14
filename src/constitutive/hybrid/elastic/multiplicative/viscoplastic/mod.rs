#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{
            plastic::Plastic,
            viscoplastic::{Viscoplastic, ViscoplasticStateVariables},
        },
        hybrid::Multiplicative,
        solid::{
            Solid,
            elastic::Elastic,
            elastic_viscoplastic::{ElasticPlasticOrViscoplastic, ElasticViscoplastic},
        },
    },
    math::{
        ContractFirstSecondIndicesWithSecondIndicesOf, ContractSecondIndexWithFirstIndexOf, Rank2,
        Scalar, Tensor,
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffnessElastic, DeformationGradient,
        DeformationGradientPlastic, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffStressElastic,
        FirstPiolaKirchhoffTangentStiffness, FirstPiolaKirchhoffTangentStiffnessElastic,
        MandelStressElastic, SecondPiolaKirchhoffStress, SecondPiolaKirchhoffStressElastic,
        SecondPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffTangentStiffnessElastic,
    },
};

impl<C1, C2> Solid for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Plastic,
{
    fn bulk_modulus(&self) -> Scalar {
        self.0.bulk_modulus()
    }
    fn shear_modulus(&self) -> Scalar {
        self.0.shear_modulus()
    }
}

impl<C1, C2> Plastic for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Plastic,
{
    fn initial_yield_stress(&self) -> Scalar {
        self.1.initial_yield_stress()
    }
    fn hardening_slope(&self) -> Scalar {
        self.1.hardening_slope()
    }
}

impl<C1, C2, Y> Viscoplastic<Y> for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Y>,
    Y: Tensor,
{
    fn initial_state(&self) -> ViscoplasticStateVariables<Y> {
        self.1.initial_state()
    }
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        state_variables: &ViscoplasticStateVariables<Y>,
    ) -> Result<ViscoplasticStateVariables<Y>, ConstitutiveError> {
        self.1.plastic_evolution(mandel_stress, state_variables)
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.1.rate_sensitivity()
    }
    fn reference_flow_rate(&self) -> Scalar {
        self.1.reference_flow_rate()
    }
}

impl<C1, C2> ElasticPlasticOrViscoplastic for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Plastic,
{
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        self.0
            .cauchy_stress(&(deformation_gradient * deformation_gradient_p.inverse()).into())
    }
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(
            CauchyTangentStiffnessElastic::from(self.0.cauchy_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?) * deformation_gradient_p_inverse.transpose(),
        )
    }
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(
            FirstPiolaKirchhoffStressElastic::from(self.0.first_piola_kirchhoff_stress(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?) * deformation_gradient_p_inverse.transpose(),
        )
    }
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        let deformation_gradient_p_inverse_transpose = deformation_gradient_p_inverse.transpose();
        Ok((FirstPiolaKirchhoffTangentStiffnessElastic::from(
            self.0.first_piola_kirchhoff_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?,
        ) * &deformation_gradient_p_inverse_transpose)
            .contract_second_index_with_first_index_of(&deformation_gradient_p_inverse_transpose))
    }
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(&deformation_gradient_p_inverse
            * SecondPiolaKirchhoffStressElastic::from(self.0.second_piola_kirchhoff_stress(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?)
            * deformation_gradient_p_inverse.transpose())
    }
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok((SecondPiolaKirchhoffTangentStiffnessElastic::from(
            self.0.second_piola_kirchhoff_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?,
        ) * deformation_gradient_p_inverse.transpose())
        .contract_first_second_indices_with_second_indices_of(
            &deformation_gradient_p_inverse,
            &deformation_gradient_p_inverse,
        ))
    }
}

impl<C1, C2, Y> ElasticViscoplastic<Y> for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic<Y>,
    Y: Tensor,
{
}
