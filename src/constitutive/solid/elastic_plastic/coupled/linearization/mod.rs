use super::ElasticPlastic;
use crate::{
    constitutive::ConstitutiveError,
    math::{ContractThirdFourthWithFirstSecond, Intermediate, Rank2, Reference},
    mechanics::{
        DeformationGradient, DeformationGradientElastic, DeformationGradientGeneral,
        DeformationGradientPlastic, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
        MandelStressElastic,
    },
};

/// The elastic-plastic stress and its tangent at fixed plastic deformation gradient,
/// from which the derivatives of the first Piola-Kirchhoff and Mandel stresses along
/// any $`(\mathrm{d}\mathbf{F},\mathrm{d}\mathbf{F}_\mathrm{p})`$ follow.
///
/// The model's Mandel stress carries $`\det\mathbf{F}`$, whereas
/// $`\mathbf{F}_\mathrm{e}^T\mathbf{P}\mathbf{F}_\mathrm{p}^T`$ carries
/// $`\det\mathbf{F}_\mathrm{e}`$; the two differ by $`\det\mathbf{F}_\mathrm{p}`$,
/// which the last step of `mandel_derivative` accounts for.
pub(super) struct Linearization {
    pub(super) tangent: FirstPiolaKirchhoffTangentStiffness,
    pub(super) stress: FirstPiolaKirchhoffStress,
    f_p: DeformationGradientPlastic,
    f_p_inverse: DeformationGradientGeneral<Reference, Intermediate>,
    f_e: DeformationGradientElastic,
    mandel: MandelStressElastic,
}

impl Linearization {
    pub(super) fn new<C: ElasticPlastic>(
        model: &C,
        f: &DeformationGradient,
        f_p: &DeformationGradientPlastic,
    ) -> Result<Self, ConstitutiveError> {
        let f_p_inverse = f_p.inverse();
        let stress = model.first_piola_kirchhoff_stress(f, f_p)?;
        let f_e = f * &f_p_inverse;
        let mandel = f_e.transpose() * &stress * f_p.transpose();
        Ok(Self {
            tangent: model.first_piola_kirchhoff_tangent_stiffness(f, f_p)?,
            stress,
            f_p: f_p.clone(),
            f_p_inverse,
            f_e,
            mandel,
        })
    }

    pub(super) fn stress_derivative(
        &self,
        d_f: &DeformationGradient,
        d_f_p: &DeformationGradientPlastic,
    ) -> FirstPiolaKirchhoffStress {
        let by_f = (&self.tangent).contract_third_fourth_with_first_second(d_f);
        let by_f_p = (&self.tangent).contract_third_fourth_with_first_second(&(&self.f_e * d_f_p));
        let correction = &self.stress * d_f_p.transpose() * self.f_p_inverse.transpose();
        by_f - by_f_p - correction
    }

    pub(super) fn mandel_derivative(
        &self,
        d_f: &DeformationGradient,
        d_f_p: &DeformationGradientPlastic,
    ) -> MandelStressElastic {
        let d_f_e = d_f * &self.f_p_inverse - &self.f_e * d_f_p * &self.f_p_inverse;
        let d_p = self.stress_derivative(d_f, d_f_p);
        let (f_e_t, f_p_t) = (self.f_e.transpose(), self.f_p.transpose());
        let d_m = d_f_e.transpose() * &self.stress * &f_p_t
            + &f_e_t * &d_p * &f_p_t
            + &f_e_t * &self.stress * d_f_p.transpose();
        let d_m = d_m + &self.mandel * (&self.f_p_inverse * d_f_p).trace();
        d_m * self.f_p.determinant()
    }
}
