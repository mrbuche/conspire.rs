use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{plastic::Plastic, viscoplastic::Viscoplastic},
        hybrid::{Hybrid, Multiplicative},
        solid::{Solid, elastic::Elastic, elastic_viscoplastic::ElasticViscoplastic},
    },
    math::{Rank2, Scalar},
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffStressElastic,
        FirstPiolaKirchhoffTangentStiffness, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffStressElastic, SecondPiolaKirchhoffTangentStiffness,
        CauchyTangentStiffnessElastic,
    },
};

// impl<C1, C2> Solid for Multiplicative<C1, C2>
// where
//     C1: Elastic,
//     C2: Viscoplastic,
// {
//     fn bulk_modulus(&self) -> &Scalar {
//         self.constitutive_model_1().bulk_modulus()
//     }
//     fn shear_modulus(&self) -> &Scalar {
//         self.constitutive_model_1().shear_modulus()
//     }
// }

impl<C1, C2> Plastic for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic,
{
    fn initial_yield_stress(&self) -> &Scalar {
        self.constitutive_model_2().initial_yield_stress()
    }
    fn hardening_slope(&self) -> &Scalar {
        self.constitutive_model_2().hardening_slope()
    }
}

impl<C1, C2> Viscoplastic for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic,
{
    fn rate_sensitivity(&self) -> &Scalar {
        self.constitutive_model_2().rate_sensitivity()
    }
    fn reference_flow_rate(&self) -> &Scalar {
        self.constitutive_model_2().reference_flow_rate()
    }
}

impl<C1, C2> ElasticViscoplastic for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Viscoplastic,
{
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        self.constitutive_model_1()
            .cauchy_stress(&(deformation_gradient * deformation_gradient_p.inverse()).into())
    }
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(CauchyTangentStiffnessElastic::from(
            self.constitutive_model_1().cauchy_tangent_stiffness(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?,
        ) * deformation_gradient_p_inverse.transpose())
    }
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(FirstPiolaKirchhoffStressElastic::from(
            self.constitutive_model_1().first_piola_kirchhoff_stress(
                &(deformation_gradient * &deformation_gradient_p_inverse).into(),
            )?,
        ) * deformation_gradient_p_inverse.transpose())
    }
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        // \mathcal{C}_{iJkL} = \mathcal{C}^\mathrm{e}_{iMkN} F_{MJ}^{\mathrm{p}-T} F_{NL}^{\mathrm{p}-T}
        todo!()
    }
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let deformation_gradient_p_inverse = deformation_gradient_p.inverse();
        Ok(&deformation_gradient_p_inverse
            * SecondPiolaKirchhoffStressElastic::from(
                self.constitutive_model_1().second_piola_kirchhoff_stress(
                    &(deformation_gradient * &deformation_gradient_p_inverse).into(),
                )?,
            )
            * deformation_gradient_p_inverse.transpose())
    }
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        // \mathcal{G}_{IJkL} = \mathcal{G}^\mathrm{e}_{MNkO} F_{MI}^{\mathrm{p}-T} F_{NJ}^{\mathrm{p}-T} F_{OL}^{\mathrm{p}-T}
        todo!()
    }
}
