//! Viscoelastic constitutive models.
//!
//! ---
//!
//! Viscoelastic constitutive models cannot be defined by a Helmholtz free energy density function and viscous dissipation function.
//! These constitutive models are therefore defined by a relation for the stress as a function of the deformation gradient and rate.
//! Consequently, the rate tangent stiffness associated with the first Piola-Kirchhoff stress is not symmetric for these models.
//!
//! ```math
//! \mathcal{U}_{iJkL} \neq \mathcal{U}_{kLiJ}
//! ```

#[cfg(test)]
pub mod test;

use super::{super::fluid::viscous::Viscous, *};
use crate::math::{
    Matrix, TensorVec, Vector,
    integrate::{Explicit, IntegrationError},
    optimize::{EqualityConstraint, FirstOrderRootFinding, OptimizeError},
};

/// Possible applied loads.
pub enum AppliedLoad<'a> {
    /// Uniaxial stress given $`\dot{F}_{11}`$.
    UniaxialStress(fn(Scalar) -> Scalar, &'a [Scalar]),
    /// Biaxial stress given $`\dot{F}_{11}`$ and $`\dot{F}_{22}`$.
    BiaxialStress(fn(Scalar) -> Scalar, fn(Scalar) -> Scalar, &'a [Scalar]),
}

/// Required methods for viscoelastic constitutive models.
pub trait Viscoelastic
where
    Self: Solid + Viscous,
{
    /// Calculates and returns the Cauchy stress.
    ///
    /// ```math
    /// \boldsymbol{\sigma} = J^{-1}\mathbf{P}\cdot\mathbf{F}^T
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyStress, ConstitutiveError> {
        Ok(deformation_gradient
            * self
                .second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_rate)?
            * deformation_gradient.transpose()
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the rate tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{V}_{ijkL} = \frac{\partial\sigma_{ij}}{\partial\dot{F}_{kL}} = J^{-1} \mathcal{W}_{MNkL} F_{iM} F_{jN}
    /// ```
    fn cauchy_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyRateTangentStiffness, ConstitutiveError> {
        Ok(self
            .second_piola_kirchhoff_rate_tangent_stiffness(
                deformation_gradient,
                deformation_gradient_rate,
            )?
            .contract_first_second_indices_with_second_indices_of(
                deformation_gradient,
                deformation_gradient,
            )
            / deformation_gradient.determinant())
    }
    /// Calculates and returns the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{P} = J\boldsymbol{\sigma}\cdot\mathbf{F}^{-T}
    /// ```
    fn first_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError> {
        Ok(
            self.cauchy_stress(deformation_gradient, deformation_gradient_rate)?
                * deformation_gradient.inverse_transpose()
                * deformation_gradient.determinant(),
        )
    }
    /// Calculates and returns the rate tangent stiffness associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{U}_{iJkL} = \frac{\partial P_{iJ}}{\partial\dot{F}_{kL}} = J \mathcal{V}_{iskL} F_{sJ}^{-T}
    /// ```
    fn first_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        Ok(self
            .cauchy_rate_tangent_stiffness(deformation_gradient, deformation_gradient_rate)?
            .contract_second_index_with_first_index_of(&deformation_gradient.inverse_transpose())
            * deformation_gradient.determinant())
    }
    /// Calculates and returns the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathbf{S} = \mathbf{F}^{-1}\cdot\mathbf{P}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        Ok(deformation_gradient.inverse()
            * self.cauchy_stress(deformation_gradient, deformation_gradient_rate)?
            * deformation_gradient.inverse_transpose()
            * deformation_gradient.determinant())
    }
    /// Calculates and returns the rate tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{W}_{IJkL} = \frac{\partial S_{IJ}}{\partial\dot{F}_{kL}} = \mathcal{U}_{mJkL}F_{mI}^{-T} = J \mathcal{V}_{mnkL} F_{mI}^{-T} F_{nJ}^{-T}
    /// ```
    fn second_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse = deformation_gradient.inverse();
        Ok(self
            .cauchy_rate_tangent_stiffness(deformation_gradient, deformation_gradient_rate)?
            .contract_first_second_indices_with_second_indices_of(
                &deformation_gradient_inverse,
                &deformation_gradient_inverse,
            )
            * deformation_gradient.determinant())
    }
    /// Solve for the unknown components of the deformation gradient and rate under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\dot{\mathbf{F}}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl Explicit<DeformationGradientRate, DeformationGradientRates>,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), IntegrationError> {
        let mut solution = DeformationGradientRate::zero();
        match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_rate_11, time) => integrator
                .integrate(
                    |t: Scalar, deformation_gradient: &DeformationGradient| {
                        solution = self.root_uniaxial_inner(
                            deformation_gradient,
                            deformation_gradient_rate_11(t),
                            &solver,
                            &solution,
                        )?;
                        Ok(solution.clone())
                    },
                    time,
                    DeformationGradient::identity(),
                ),
            AppliedLoad::BiaxialStress(
                deformation_gradient_rate_11,
                deformation_gradient_rate_22,
                time,
            ) => integrator.integrate(
                |t: Scalar, deformation_gradient: &DeformationGradient| {
                    solution = self.root_biaxial_inner(
                        deformation_gradient,
                        deformation_gradient_rate_11(t),
                        deformation_gradient_rate_22(t),
                        &solver,
                        &solution,
                    )?;
                    Ok(solution.clone())
                },
                time,
                DeformationGradient::identity(),
            ),
        }
    }
    #[doc(hidden)]
    fn root_uniaxial_inner(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate_11: Scalar,
        solver: &impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
        initial_guess: &DeformationGradientRate,
    ) -> Result<DeformationGradientRate, OptimizeError> {
        let mut matrix = Matrix::zero(4, 9);
        let mut vector = Vector::zero(4);
        matrix[0][0] = 1.0;
        matrix[1][1] = 1.0;
        matrix[2][2] = 1.0;
        matrix[3][5] = 1.0;
        vector[0] = deformation_gradient_rate_11;
        solver.root(
            |deformation_gradient_rate: &DeformationGradientRate| {
                Ok(self.first_piola_kirchhoff_stress(
                    deformation_gradient,
                    deformation_gradient_rate,
                )?)
            },
            |deformation_gradient_rate: &DeformationGradientRate| {
                Ok(self.first_piola_kirchhoff_rate_tangent_stiffness(
                    deformation_gradient,
                    deformation_gradient_rate,
                )?)
            },
            initial_guess.clone(),
            EqualityConstraint::Linear(matrix, vector),
        )
    }
    #[doc(hidden)]
    fn root_biaxial_inner(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate_11: Scalar,
        deformation_gradient_rate_22: Scalar,
        solver: &impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffTangentStiffness,
            DeformationGradient,
        >,
        initial_guess: &DeformationGradientRate,
    ) -> Result<DeformationGradientRate, OptimizeError> {
        let mut matrix = Matrix::zero(5, 9);
        let mut vector = Vector::zero(5);
        matrix[0][0] = 1.0;
        matrix[1][1] = 1.0;
        matrix[2][2] = 1.0;
        matrix[3][5] = 1.0;
        matrix[4][4] = 1.0;
        vector[0] = deformation_gradient_rate_11;
        vector[4] = deformation_gradient_rate_22;
        solver.root(
            |deformation_gradient_rate: &DeformationGradientRate| {
                Ok(self.first_piola_kirchhoff_stress(
                    deformation_gradient,
                    deformation_gradient_rate,
                )?)
            },
            |deformation_gradient_rate: &DeformationGradientRate| {
                Ok(self.first_piola_kirchhoff_rate_tangent_stiffness(
                    deformation_gradient,
                    deformation_gradient_rate,
                )?)
            },
            initial_guess.clone(),
            EqualityConstraint::Linear(matrix, vector),
        )
    }
}
