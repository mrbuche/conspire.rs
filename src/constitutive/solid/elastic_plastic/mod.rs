//! Elastic-plastic solid constitutive models.

use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::Plastic, solid::Solid},
    math::{
        ContractFirstSecondWithSecond, ContractSecondWithFirst, IDENTITY, Matrix, Quantity, Rank2,
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic,
        FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness, MandelStressElastic,
        Scalar, SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
    },
    units::Time,
};

/// Possible applied loads.
pub enum AppliedLoad<'a> {
    /// Uniaxial stress given $`F_{11}`$.
    UniaxialStress(fn(Quantity<Time>) -> Scalar, &'a [Quantity<Time>]),
    /// Biaxial stress given $`F_{11}`$ and $`F_{22}`$.
    BiaxialStress(
        fn(Quantity<Time>) -> Scalar,
        fn(Quantity<Time>) -> Scalar,
        &'a [Quantity<Time>],
    ),
}

type Prescribed = Vec<(usize, fn(Quantity<Time>) -> Scalar)>;

#[doc(hidden)]
pub fn bcs(applied_load: AppliedLoad<'_>) -> (Matrix, Prescribed, &'_ [Quantity<Time>]) {
    let (mut matrix, prescribed, time) = match applied_load {
        AppliedLoad::UniaxialStress(deformation_gradient_11, time) => {
            (Matrix::zero(4, 9), vec![(0, deformation_gradient_11)], time)
        }
        AppliedLoad::BiaxialStress(deformation_gradient_11, deformation_gradient_22, time) => (
            Matrix::zero(5, 9),
            vec![(0, deformation_gradient_11), (4, deformation_gradient_22)],
            time,
        ),
    };
    matrix[0][0] = 1.0;
    matrix[1][1] = 1.0;
    matrix[2][2] = 1.0;
    matrix[3][5] = 1.0;
    if matrix.len() == 5 {
        matrix[4][4] = 1.0
    }
    (matrix, prescribed, time)
}

/// Required methods for elastic-plastic or elastic-viscoplastic solid constitutive models.
pub trait ElasticPlasticOrViscoplastic
where
    Self: Solid + Plastic,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma} = \boldsymbol{\sigma}_\mathrm{e}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(deformation_gradient
            * self.second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\mathcal{T}} = \boldsymbol{\mathcal{T}}_\mathrm{e}\cdot\mathbf{F}_\mathrm{p}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        let some_stress = &cauchy_stress * &deformation_gradient_inverse_transpose;
        Ok(self
            .second_piola_kirchhoff_tangent_stiffness(deformation_gradient, deformation_gradient_p)?
            .contract_first_second_with_second(deformation_gradient, deformation_gradient)
            / deformation_gradient.determinant()
            - CauchyTangentStiffness::dyad_ij_kl(
                &cauchy_stress,
                &deformation_gradient_inverse_transpose,
            )
            + CauchyTangentStiffness::dyad_il_kj(&some_stress, &IDENTITY)
            + CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, &some_stress))
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P} = \mathbf{P}_\mathrm{e}\cdot\mathbf{F}_\mathrm{p}^{-T}
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(
            self.cauchy_stress(deformation_gradient, deformation_gradient_p)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the tangent stiffness associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{C}_{iJkL} = \mathcal{C}^\mathrm{e}_{iMkN} F_{MJ}^{\mathrm{p}-T} F_{NL}^{\mathrm{p}-T}
    /// ```
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let first_piola_kirchhoff_stress =
            self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_p)?
            .contract_second_with_first(&deformation_gradient_inverse_transpose)
            * deformation_gradient.determinant()
            + FirstPiolaKirchhoffTangentStiffness::dyad_ij_kl(
                &first_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - FirstPiolaKirchhoffTangentStiffness::dyad_il_kj(
                &first_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            ))
    }
    /// Calculates and returns the Mandel stress.
    ///
    /// ```math
    /// \mathbf{M}_\mathrm{e} = J\mathbf{F}_\mathrm{e}^T\cdot\boldsymbol{\sigma}\cdot\mathbf{F}_\mathrm{e}^{-T}
    /// ```
    fn mandel_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<MandelStressElastic, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let deformation_gradient_e = deformation_gradient * deformation_gradient_p.inverse();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_p)?;
        Ok((deformation_gradient_e.transpose()
            * cauchy_stress
            * deformation_gradient_e.inverse_transpose())
            * jacobian)
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}_\mathrm{p}^{-1}\cdot\mathbf{S}_\mathrm{e}\cdot\mathbf{F}_\mathrm{p}^{-T}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?)
    }
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}_{IJkL} = \mathcal{G}^\mathrm{e}_{MNkO} F_{MI}^{\mathrm{p}-T} F_{NJ}^{\mathrm{p}-T} F_{OL}^{\mathrm{p}-T}
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_p: &DeformationGradientPlastic,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let deformation_gradient_inverse = deformation_gradient_inverse_transpose.transpose();
        let second_piola_kirchhoff_stress =
            self.second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_p)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_p)?
            .contract_first_second_with_second(
                &deformation_gradient_inverse,
                &deformation_gradient_inverse,
            )
            * deformation_gradient.determinant()
            + SecondPiolaKirchhoffTangentStiffness::dyad_ij_kl(
                &second_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchhoffTangentStiffness::dyad_il_kj(
                &second_piola_kirchhoff_stress,
                &deformation_gradient_inverse_transpose,
            )
            - SecondPiolaKirchhoffTangentStiffness::dyad_ik_jl(
                &deformation_gradient_inverse,
                &second_piola_kirchhoff_stress,
            ))
    }
}

/// Required methods for elastic-plastic solid constitutive models.
pub trait ElasticPlastic
where
    Self: ElasticPlasticOrViscoplastic,
{
}
