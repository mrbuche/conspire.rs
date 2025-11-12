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
        IDENTITY_10, Rank2, TensorArray, TensorRank2,
        optimize::{EqualityConstraint, GradientDescent, ZerothOrderRootFinding},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, CauchyTangentStiffness1, DeformationGradient,
        DeformationGradient2, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffStress1,
        FirstPiolaKirchhoffStress2, FirstPiolaKirchhoffTangentStiffness, Scalar,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
    },
};

impl<C1, C2> Solid for Multiplicative<C1, C2>
where
    // C1: Elastic,
    // C2: Elastic,
    C1: Solid,
    C2: std::fmt::Debug,
{
    fn bulk_modulus(&self) -> Scalar {
        todo!()
    }
    fn shear_modulus(&self) -> Scalar {
        todo!()
    }
}

impl<C1, C2> ElasticIV<DeformationGradient2> for Multiplicative<C1, C2>
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
    fn internal_variables_initial_value(&self) -> DeformationGradient2 {
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
                .first_piola_kirchhoff_stress(&deformation_gradient_2.clone().into())?,
        ) - deformation_gradient_1.transpose()
            * FirstPiolaKirchhoffStress1::from(
                self.0
                    .first_piola_kirchhoff_stress(&deformation_gradient_1.into())?,
            )
            * deformation_gradient_2_inverse.transpose())
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
