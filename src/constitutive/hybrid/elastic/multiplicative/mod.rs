#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        hybrid::{Multiplicative, MultiplicativeTrait},
        solid::{
            Solid,
            elastic::{Elastic, internal_variables::ElasticIV},
        },
    },
    math::{
        ContractThirdIndexWithFirstIndexOf, IDENTITY_10, Rank2, TensorArray, TensorRank2,
        TensorRank4,
        optimize::{EqualityConstraint, GradientDescent, ZerothOrderRootFinding},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffness1, DeformationGradient,
        DeformationGradient2, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffStress1,
        FirstPiolaKirchhoffStress2, FirstPiolaKirchhoffTangentStiffness,
        FirstPiolaKirchhoffTangentStiffness2, Scalar, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
};

impl<C1, C2> Solid for Multiplicative<C1, C2>
where
    // C1: Elastic,
    // C2: Elastic,
    C1: Solid,
    C2: Clone + std::fmt::Debug,
{
    fn bulk_modulus(&self) -> Scalar {
        todo!() // can do right when switch to the _foo methods
    }
    fn shear_modulus(&self) -> Scalar {
        todo!() // can do right when switch to the _foo methods
    }
}

impl<C1, C2>
    ElasticIV<
        DeformationGradient2,
        TensorRank4<3, 2, 0, 1, 0>,
        TensorRank4<3, 1, 0, 2, 0>,
        FirstPiolaKirchhoffTangentStiffness2,
    > for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma} = \frac{1}{J_2}\,\boldsymbol{\sigma}_1
    /// ```
    fn cauchy_stress_foo(
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
    fn cauchy_tangent_stiffness_foo(
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
    fn first_piola_kirchhoff_stress_foo(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(
            self.cauchy_stress_foo(deformation_gradient, deformation_gradient_2)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}_2^{-1}\cdot\mathbf{S}_1\cdot\mathbf{F}_2^{-T}
    /// ```
    fn second_piola_kirchhoff_stress_foo(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self
                .first_piola_kirchhoff_stress_foo(deformation_gradient, deformation_gradient_2)?)
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
    /// Calculates and returns the tangents associated with the internal variables.
    ///
    /// ```math
    /// \frac{\partial P_{iJ}}{\partial F_{KL}^2} = -P_{iL}F_{KJ}^{2-T} - \mathcal{C}_{iJmL}F_{mK}^1
    /// ```
    /// ```math
    /// \frac{\partial R_{IJ}}{\partial F_{kL}} = -F_{IL}^{2-T}P_{kJ} - F_{mI}^1\mathcal{C}_{mJkL}
    /// ```
    /// ```math
    /// \frac{\partial R_{IJ}}{\partial F_{KL}^2} = \mathcal{C}_{IJKL}^2 + F_{IM}^1P_{ML}{F_{KJ}^{2-T}} - \frac{\partial R_{IJ}}{\partial F_{mL}}\,F_{mK}^1
    /// ```
    fn internal_variables_tangents(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_2: &DeformationGradient2,
    ) -> Result<
        (
            TensorRank4<3, 2, 0, 1, 0>,
            TensorRank4<3, 1, 0, 2, 0>,
            FirstPiolaKirchhoffTangentStiffness2,
        ),
        ConstitutiveError,
    > {
        //
        // If hyperelastic, tangent_1 should equal a transpose of tangent_2.
        // Could add a method to utilize that for minimize() in hyperelastic/multiplicative.
        //
        let deformation_gradient_2_inverse = deformation_gradient_2.inverse();
        let deformation_gradient_2_inverse_transpose = deformation_gradient_2_inverse.transpose();
        let deformation_gradient_1 = deformation_gradient * &deformation_gradient_2_inverse;
        let deformation_gradient_1_transpose = deformation_gradient_1.transpose();
        let first_piola_kirchhoff_stress =
            self.first_piola_kirchhoff_stress_foo(deformation_gradient, deformation_gradient_2)?;
        let tangent_0 = self.first_piola_kirchhoff_tangent_stiffness_foo(
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
        ) - tangent_0
            .contract_third_index_with_first_index_of(&deformation_gradient_1);
        let tangent_3 = FirstPiolaKirchhoffTangentStiffness2::from(
            self.1
                .first_piola_kirchhoff_tangent_stiffness(deformation_gradient_2.into())?,
        ) - tangent_1
            .contract_third_index_with_first_index_of(&deformation_gradient_1)
            + TensorRank4::dyad_il_jk(
                &(deformation_gradient_1_transpose * first_piola_kirchhoff_stress),
                &deformation_gradient_2_inverse,
            );
        Ok((tangent_1, tangent_2, tangent_3))
    }
    fn internal_variables_constraints(&self) -> (&[usize], usize) {
        (&[10, 11, 14], 9)
    }
}

impl<C1, C2> Elastic for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{1}{J_2}\,\boldsymbol{\sigma}_1(\mathbf{F}_1)
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let (deformation_gradient_1, deformation_gradient_2) =
            self.deformation_gradients(deformation_gradient)?;
        Ok(self.0.cauchy_stress(&deformation_gradient_1)? / deformation_gradient_2.determinant())
    }
    /// Dummy method that will panic.
    fn cauchy_tangent_stiffness(
        &self,
        _: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        panic!()
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F}) = \mathbf{P}_1(\mathbf{F}_1)\cdot\mathbf{F}_2^{-T}
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        let (deformation_gradient_1, deformation_gradient_2) =
            self.deformation_gradients(deformation_gradient)?;
        let deformation_gradient_2_inverse_transpose: TensorRank2<3, 0, 0> =
            deformation_gradient_2.inverse_transpose().into();
        Ok(self
            .0
            .first_piola_kirchhoff_stress(&deformation_gradient_1)?
            * deformation_gradient_2_inverse_transpose)
    }
    /// Dummy method that will panic.
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        _: &DeformationGradient,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        panic!()
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S}(\mathbf{F}) = \mathbf{F}_2^{-1}\cdot\mathbf{S}_1(\mathbf{F}_1)\cdot\mathbf{F}_2^{-T}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let (deformation_gradient_1, deformation_gradient_2) =
            self.deformation_gradients(deformation_gradient)?;
        let deformation_gradient_2_inverse: TensorRank2<3, 0, 0> =
            deformation_gradient_2.inverse().into();
        Ok(&deformation_gradient_2_inverse
            * self
                .0
                .second_piola_kirchhoff_stress(&deformation_gradient_1)?
            * deformation_gradient_2_inverse.transpose())
    }
    /// Dummy method that will panic.
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        _: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        panic!()
    }
}

impl<C1, C2> MultiplicativeTrait for Multiplicative<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    fn deformation_gradients(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<(DeformationGradient, DeformationGradient), ConstitutiveError> {
        if deformation_gradient.is_identity() {
            Ok((IDENTITY_10, IDENTITY_10))
        } else {
            match GradientDescent::default().root(
                |deformation_gradient_2: &DeformationGradient| {
                    let deformation_gradient_1: DeformationGradient =
                        (deformation_gradient * deformation_gradient_2.inverse()).into();
                    let deformation_gradient_2_inverse_transpose: TensorRank2<3, 0, 0> =
                        deformation_gradient_2.inverse_transpose().into();
                    let right_hand_side: FirstPiolaKirchhoffStress = (deformation_gradient_1
                        .transpose()
                        * self
                            .0
                            .first_piola_kirchhoff_stress(&deformation_gradient_1)?
                        * deformation_gradient_2_inverse_transpose)
                        .into();
                    Ok(self
                        .1
                        .first_piola_kirchhoff_stress(deformation_gradient_2)?
                        - right_hand_side)
                },
                IDENTITY_10,
                EqualityConstraint::None,
            ) {
                Ok(deformation_gradient_2) => {
                    let deformation_gradient_1 =
                        (deformation_gradient * deformation_gradient_2.inverse()).into();
                    Ok((deformation_gradient_1, deformation_gradient_2))
                }
                Err(error) => Err(ConstitutiveError::Upstream(
                    format!("{error}"),
                    format!("{self:?}"),
                )),
            }
        }
    }
}
