#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        hybrid::{Hybrid, Multiplicative, MultiplicativeTrait},
        solid::{Solid, elastic::Elastic},
    },
    math::{
        IDENTITY_10, Rank2, TensorRank2,
        optimize::{EqualityConstraint, GradientDescent, ZerothOrderRootFinding},
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, FirstPiolaKirchhoffStress,
        FirstPiolaKirchhoffTangentStiffness, Scalar, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
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
        Ok(self
            .constitutive_model_1()
            .cauchy_stress(&deformation_gradient_1)?
            / deformation_gradient_2.determinant())
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
            .constitutive_model_1()
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
                .constitutive_model_1()
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
                            .constitutive_model_1()
                            .first_piola_kirchhoff_stress(&deformation_gradient_1)?
                        * deformation_gradient_2_inverse_transpose)
                        .into();
                    Ok(self
                        .constitutive_model_2()
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
