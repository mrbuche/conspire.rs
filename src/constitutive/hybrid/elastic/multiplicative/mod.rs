#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        hybrid::{Hybrid, Multiplicative, MultiplicativeTrait},
        solid::{elastic::Elastic, Solid},
        Constitutive, ConstitutiveError, Parameters,
    },
    math::{Rank2, Tensor, TensorRank2},
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchoffStress,
        FirstPiolaKirchoffTangentStiffness, Scalar, SecondPiolaKirchoffStress,
        SecondPiolaKirchoffTangentStiffness, IDENTITY_10, ZERO_10,
    },
    ABS_TOL,
};

impl<'a, C1: Elastic<'a>, C2: Elastic<'a>> Constitutive<'a> for Multiplicative<C1, C2> {
    /// Dummy method that will panic, use [Self::construct()] instead.
    fn new(_parameters: Parameters<'a>) -> Self {
        panic!()
    }
}

impl<'a, C1: Elastic<'a>, C2: Elastic<'a>> Solid<'a> for Multiplicative<C1, C2> {
    /// Dummy method that will panic.
    fn get_bulk_modulus(&self) -> &Scalar {
        panic!()
    }
    /// Dummy method that will panic.
    fn get_shear_modulus(&self) -> &Scalar {
        panic!()
    }
}

impl<'a, C1: Elastic<'a>, C2: Elastic<'a>> Elastic<'a> for Multiplicative<C1, C2> {
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{1}{J_2}\,\boldsymbol{\sigma}_1(\mathbf{F}_1)
    /// ```
    fn calculate_cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let (deformation_gradient_1, deformation_gradient_2) =
            self.calculate_deformation_gradients(deformation_gradient)?;
        Ok(self
            .get_constitutive_model_1()
            .calculate_cauchy_stress(&deformation_gradient_1)?
            / deformation_gradient_2.determinant())
    }
    /// Dummy method that will panic.
    fn calculate_cauchy_tangent_stiffness(
        &self,
        _: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        panic!()
    }
    /// Calculates and returns the first Piola-Kirchoff stress.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F}) = \mathbf{P}_1(\mathbf{F}_1)\cdot\mathbf{F}_2^{-T}
    /// ```
    fn calculate_first_piola_kirchoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<FirstPiolaKirchoffStress, ConstitutiveError> {
        let (deformation_gradient_1, deformation_gradient_2) =
            self.calculate_deformation_gradients(deformation_gradient)?;
        let deformation_gradient_2_inverse_transpose: TensorRank2<3, 0, 0> =
            deformation_gradient_2.inverse_transpose().into();
        Ok(self
            .get_constitutive_model_1()
            .calculate_first_piola_kirchoff_stress(&deformation_gradient_1)?
            * deformation_gradient_2_inverse_transpose)
    }
    /// Dummy method that will panic.
    fn calculate_first_piola_kirchoff_tangent_stiffness(
        &self,
        _: &DeformationGradient,
    ) -> Result<FirstPiolaKirchoffTangentStiffness, ConstitutiveError> {
        panic!()
    }
    /// Calculates and returns the second Piola-Kirchoff stress.
    ///
    /// ```math
    /// \mathbf{S}(\mathbf{F}) = \mathbf{F}_2^{-1}\cdot\mathbf{S}_1(\mathbf{F}_1)\cdot\mathbf{F}_2^{-T}
    /// ```
    fn calculate_second_piola_kirchoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchoffStress, ConstitutiveError> {
        let (deformation_gradient_1, deformation_gradient_2) =
            self.calculate_deformation_gradients(deformation_gradient)?;
        let deformation_gradient_2_inverse: TensorRank2<3, 0, 0> =
            deformation_gradient_2.inverse().into();
        Ok(&deformation_gradient_2_inverse
            * self
                .get_constitutive_model_1()
                .calculate_second_piola_kirchoff_stress(&deformation_gradient_1)?
            * deformation_gradient_2_inverse.transpose())
    }
    /// Dummy method that will panic.
    fn calculate_second_piola_kirchoff_tangent_stiffness(
        &self,
        _: &DeformationGradient,
    ) -> Result<SecondPiolaKirchoffTangentStiffness, ConstitutiveError> {
        panic!()
    }
}

impl<'a, C1: Elastic<'a>, C2: Elastic<'a>> MultiplicativeTrait for Multiplicative<C1, C2> {
    fn calculate_deformation_gradients(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<(DeformationGradient, DeformationGradient), ConstitutiveError> {
        if deformation_gradient.is_identity() {
            Ok((IDENTITY_10, IDENTITY_10))
        } else {
            let mut deformation_gradient_1 = IDENTITY_10;
            let mut deformation_gradient_2 = deformation_gradient.copy();
            let mut deformation_gradient_2_old = IDENTITY_10;
            let mut deformation_gradient_2_inverse_transpose: TensorRank2<3, 0, 0>;
            let mut residual: FirstPiolaKirchoffStress;
            let mut residual_increment: FirstPiolaKirchoffStress;
            let mut residual_norm = 1.0;
            let mut residual_old = ZERO_10;
            let mut right_hand_side: FirstPiolaKirchoffStress;
            let mut steps: u8 = 0;
            let steps_maximum: u8 = 50;
            let mut step_size: Scalar;
            while steps < steps_maximum {
                deformation_gradient_1 =
                    (deformation_gradient * deformation_gradient_2.inverse()).into();
                deformation_gradient_2_inverse_transpose =
                    deformation_gradient_2.inverse_transpose().into();
                right_hand_side = (deformation_gradient_1.transpose()
                    * self
                        .get_constitutive_model_1()
                        .calculate_first_piola_kirchoff_stress(&deformation_gradient_1)?
                    * deformation_gradient_2_inverse_transpose)
                    .into();
                residual = self
                    .get_constitutive_model_2()
                    .calculate_first_piola_kirchoff_stress(&deformation_gradient_2)?
                    - right_hand_side;
                residual_norm = residual.norm();
                if residual_norm >= ABS_TOL {
                    residual_increment = residual_old - &residual;
                    step_size = (deformation_gradient_2_old - &deformation_gradient_2)
                        .full_contraction(&residual_increment)
                        .abs()
                        / residual_increment.norm_squared();
                    deformation_gradient_2_old = deformation_gradient_2.copy();
                    residual_old = residual.copy();
                    deformation_gradient_2 -= residual * step_size;
                } else {
                    break;
                }
                steps += 1;
            }
            if residual_norm >= ABS_TOL && steps == steps_maximum {
                panic!("MAX STEPS REACHED")
            } else {
                Ok((deformation_gradient_1, deformation_gradient_2))
            }
        }
    }
}
