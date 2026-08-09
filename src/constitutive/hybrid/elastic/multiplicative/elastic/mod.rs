#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        hybrid::ElasticMultiplicative,
        solid::{
            Solid,
            elastic::{Elastic, internal_variables::ElasticIV},
        },
    },
    math::{ContractThirdWithFirst, Rank2, TensorArray, TensorRank4},
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffness1, DeformationGradient,
        DeformationGradient2, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffStress1,
        FirstPiolaKirchhoffStress2, FirstPiolaKirchhoffTangentStiffness,
        FirstPiolaKirchhoffTangentStiffness2, Scalar, SecondPiolaKirchhoffStress,
    },
};

impl<C1, C2> Solid for ElasticMultiplicative<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    fn bulk_modulus(&self) -> Scalar {
        1.0 / (1.0 / self.0.bulk_modulus() + 1.0 / self.1.bulk_modulus())
    }
    fn shear_modulus(&self) -> Scalar {
        1.0 / (1.0 / self.0.shear_modulus() + 1.0 / self.1.shear_modulus())
    }
}

impl<C1, C2> ElasticIV<DeformationGradient2> for ElasticMultiplicative<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    type TangentVu = TensorRank4<3, 2, 0, 1, 0>;
    type TangentUv = TensorRank4<3, 1, 0, 2, 0>;
    type TangentVv = FirstPiolaKirchhoffTangentStiffness2;
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma} = \frac{1}{J_2}\,\boldsymbol{\sigma}_1
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let (deformation_gradient_2_inverse, jacobian_2) =
            deformation_gradient_2.inverse_and_determinant();
        let deformation_gradient_1 = deformation_gradient * &deformation_gradient_2_inverse;
        Ok(self.0.cauchy_stress(&deformation_gradient_1.into())? / jacobian_2)
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\mathcal{T}} = \frac{1}{J_2}\,\boldsymbol{\mathcal{T}}_1\cdot\mathbf{F}_2^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let (deformation_gradient_2_inverse, jacobian_2) =
            deformation_gradient_2.inverse_and_determinant();
        let deformation_gradient_1 = deformation_gradient * &deformation_gradient_2_inverse;
        Ok(CauchyTangentStiffness1::from(
            self.0
                .cauchy_tangent_stiffness(&deformation_gradient_1.into())?,
        ) * (deformation_gradient_2_inverse.transpose() / jacobian_2))
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P} = \mathbf{P}_1\cdot\mathbf{F}_2^{-T}
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(
            self.cauchy_stress(deformation_gradient, deformation_gradient_2)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}_2^{-1}\cdot\mathbf{S}_1\cdot\mathbf{F}_2^{-T}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_2)?)
    }
    fn internal_variables_initial(&self) -> DeformationGradient2 {
        DeformationGradient2::identity()
    }
    /// Calculates and returns the residual associated with the second deformation gradient.
    ///
    /// ```math
    /// \mathbf{R} = \mathbf{P}_2 - \mathbf{M}_1\cdot\mathbf{F}_2^{-T}
    /// ```
    fn internal_variables_residual(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<DeformationGradient2, ConstitutiveError> {
        let deformation_gradient_2_inverse = deformation_gradient_2.inverse();
        let deformation_gradient_1 = deformation_gradient * &deformation_gradient_2_inverse;
        Ok(FirstPiolaKirchhoffStress2::from(
            self.1
                .first_piola_kirchhoff_stress(deformation_gradient_2.into())?,
        ) - deformation_gradient_1.transpose()
            * FirstPiolaKirchhoffStress1::from(
                self.0
                    .first_piola_kirchhoff_stress(&deformation_gradient_1.into())?,
            )
            * deformation_gradient_2_inverse.transpose())
    }
    /// Calculates and returns the tangents of the coupled system.
    ///
    /// ```math
    /// \mathcal{C}_{iJkL} = \frac{\partial P_{iJ}}{\partial F_{kL}}
    /// ```
    /// ```math
    /// \frac{\partial R_{IJ}}{\partial F_{kL}} = -F_{IL}^{2-T}P_{kJ} - F_{mI}^1\mathcal{C}_{mJkL}
    /// ```
    /// ```math
    /// \frac{\partial P_{iJ}}{\partial F_{KL}^2} = -P_{iL}F_{KJ}^{2-T} - \mathcal{C}_{iJmL}F_{mK}^1
    /// ```
    /// ```math
    /// \frac{\partial R_{IJ}}{\partial F_{KL}^2} = \mathcal{C}_{IJKL}^2 + F_{IM}^1P_{ML}{F_{KJ}^{2-T}} - \frac{\partial R_{IJ}}{\partial F_{mL}}\,F_{mK}^1
    /// ```
    fn tangents(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<
        (
            FirstPiolaKirchhoffTangentStiffness,
            TensorRank4<3, 2, 0, 1, 0>,
            TensorRank4<3, 1, 0, 2, 0>,
            FirstPiolaKirchhoffTangentStiffness2,
        ),
        ConstitutiveError,
    > {
        let deformation_gradient_2_inverse = deformation_gradient_2.inverse();
        let deformation_gradient_2_inverse_transpose = deformation_gradient_2_inverse.transpose();
        let deformation_gradient_1 = deformation_gradient * &deformation_gradient_2_inverse;
        let deformation_gradient_1_transpose = deformation_gradient_1.transpose();
        let first_piola_kirchhoff_stress =
            self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_2)?;
        let tangent_0 = self.first_piola_kirchhoff_tangent_stiffness(
            deformation_gradient,
            deformation_gradient_2,
        )?;
        let tangent_1 = TensorRank4::dyad_il_kj(
            &(deformation_gradient_2_inverse_transpose * -1.0),
            &first_piola_kirchhoff_stress,
        ) - &deformation_gradient_1_transpose * &tangent_0;
        let tangent_2 = TensorRank4::dyad_il_jk(
            &first_piola_kirchhoff_stress,
            &(&deformation_gradient_2_inverse * -1.0),
        ) - tangent_0.contract_third_with_first(&deformation_gradient_1);
        let tangent_3 = FirstPiolaKirchhoffTangentStiffness2::from(
            self.1
                .first_piola_kirchhoff_tangent_stiffness(deformation_gradient_2.into())?,
        ) - tangent_1.contract_third_with_first(&deformation_gradient_1)
            + TensorRank4::dyad_il_jk(
                &(deformation_gradient_1_transpose * first_piola_kirchhoff_stress),
                &deformation_gradient_2_inverse,
            );
        Ok((tangent_0, tangent_1, tangent_2, tangent_3))
    }
    /// The strict upper triangle of the second deformation gradient, which keeps it
    /// lower triangular and thereby fixes the rotational freedom of the split.
    fn internal_variables_fixed(&self) -> &[usize] {
        &[1, 2, 5]
    }
}
