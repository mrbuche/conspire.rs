//! Viscoelastic solid constitutive models.
//!
//! ---
//!
//! Viscoelastic solid constitutive models cannot be defined by a Helmholtz free energy density and a viscous dissipation function.
//! These constitutive models are therefore defined by a relation for the stress as a function of the deformation gradient and rate.
//! Consequently, the rate tangent stiffness associated with the first Piola-Kirchhoff stress is not symmetric for these models.
//!
//! ```math
//! \mathcal{U}_{iJkL} \neq \mathcal{U}_{kLiJ}
//! ```

// Could eventually make this a bit more general with a const generic type parameters:
// (0) implicitly rate dependent (has rate-dependent ISV)
// (1) explicitly rate dependent to first order (dF/dt)
// (2) explicitly rate dependent to second order (d^2F/dt^2)
// ...
// Still need separate traits, though, since the arguments will differ.
// And will have to implement differently anyway, especially in fem/
// So maybe instead, there are just subdirectories in viscoelastic/
// order_0/
// order_1/
// order_2/
// ...
// But how to handle order_1+ that may or may not have ISV?
// Maybe just have separate trait, like ElastiicViscoplasticISV.
// But what about viscoelastic models with rate-dependent and/or rate-independent ISV?
// Maybe the key is there are *IVs* and *SVs*:
// internal variables are determined by the state (not independent), and cannot be rate-dependent,
// and state variables determine the state (are independent), and must be rate-dependent.
// The ability (or inability) to set these models up properly as
// (a) optimization (or root-finding) problems with constraints, and
// (b) ensure that the second law is satisfied (Lyapunov stability),
// should determine whether the model can fit into this framework.
// For example, if a model has rate-independent variables but is path-dependent,
// the model cannot fit into this framework, and may be invalid altogether,
// or at least that doing the model that was is a bad idea.
// That might be the key question:
// Can (a) and (b) be satisfied for rate-independent, but path-dependent, variables?
// Rate-independent elasto-plasticity being the key example.
// If (a) is true, the plastic deformation is not independent.
// But if (a) is not true, what problem are we solving?

#[cfg(test)]
pub mod test;

use super::{super::fluid::viscous::Viscous, *};
use crate::math::{
    Matrix, Vector,
    integrate::{ImplicitDaeFirstOrderRoot, ImplicitDaeZerothOrderRoot},
    optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
};

/// Possible applied loads.
pub enum AppliedLoad<'a> {
    /// Uniaxial stress given $`\dot{F}_{11}`$.
    UniaxialStress(fn(Scalar) -> Scalar, &'a [Scalar]),
    /// Biaxial stress given $`\dot{F}_{11}`$ and $`\dot{F}_{22}`$.
    BiaxialStress(fn(Scalar) -> Scalar, fn(Scalar) -> Scalar, &'a [Scalar]),
}

/// Required methods for viscoelastic solid constitutive models.
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
}

/// Zeroth-order root-finding methods for viscoelastic solid constitutive models.
pub trait ZerothOrderRoot {
    /// Solve for the unknown components of the deformation gradient and rate under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\dot{\mathbf{F}}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeZerothOrderRoot<DeformationGradient, DeformationGradients>,
        solver: impl ZerothOrderRootFinding<DeformationGradient>,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), ConstitutiveError>;
}

/// Zeroth-order root-finding methods for viscoelastic solid constitutive models.
pub trait FirstOrderRoot {
    /// Solve for the unknown components of the deformation gradient and rate under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\dot{\mathbf{F}}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeFirstOrderRoot<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradientRate,
            DeformationGradientRates,
        >,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradientRate,
        >,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), ConstitutiveError>;
}

impl<T> ZerothOrderRoot for T
where
    T: Viscoelastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeZerothOrderRoot<DeformationGradient, DeformationGradients>,
        solver: impl ZerothOrderRootFinding<DeformationGradientRate>,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), ConstitutiveError> {
        match match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_rate_11, time) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                integrator.integrate(
                    |_: Scalar,
                     deformation_gradient: &DeformationGradient,
                     deformation_gradient_rate: &DeformationGradientRate| {
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
                    solver,
                    time,
                    DeformationGradient::identity(),
                    |t: Scalar| {
                        vector[0] = deformation_gradient_rate_11(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                )
            }
            AppliedLoad::BiaxialStress(
                deformation_gradient_rate_11,
                deformation_gradient_rate_22,
                time,
            ) => {
                let mut matrix = Matrix::zero(5, 9);
                let mut vector = Vector::zero(5);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                matrix[4][4] = 1.0;
                integrator.integrate(
                    |_: Scalar,
                     deformation_gradient: &DeformationGradient,
                     deformation_gradient_rate: &DeformationGradientRate| {
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
                    solver,
                    time,
                    DeformationGradient::identity(),
                    |t: Scalar| {
                        vector[0] = deformation_gradient_rate_11(t);
                        vector[4] = deformation_gradient_rate_22(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                )
            }
        } {
            Ok(results) => Ok(results),
            Err(error) => Err(ConstitutiveError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<T> FirstOrderRoot for T
where
    T: Viscoelastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeFirstOrderRoot<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradientRate,
            DeformationGradientRates,
        >,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradientRate,
        >,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), ConstitutiveError> {
        match match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_rate_11, time) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                integrator.integrate(
                    |_: Scalar,
                     deformation_gradient: &DeformationGradient,
                     deformation_gradient_rate: &DeformationGradientRate| {
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
                    |_: Scalar,
                     deformation_gradient: &DeformationGradient,
                     deformation_gradient_rate: &DeformationGradientRate| {
                        Ok(self.first_piola_kirchhoff_rate_tangent_stiffness(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
                    solver,
                    time,
                    DeformationGradient::identity(),
                    |t: Scalar| {
                        vector[0] = deformation_gradient_rate_11(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                )
            }
            AppliedLoad::BiaxialStress(
                deformation_gradient_rate_11,
                deformation_gradient_rate_22,
                time,
            ) => {
                let mut matrix = Matrix::zero(5, 9);
                let mut vector = Vector::zero(5);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                matrix[4][4] = 1.0;
                integrator.integrate(
                    |_: Scalar,
                     deformation_gradient: &DeformationGradient,
                     deformation_gradient_rate: &DeformationGradientRate| {
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
                    |_: Scalar,
                     deformation_gradient: &DeformationGradient,
                     deformation_gradient_rate: &DeformationGradientRate| {
                        Ok(self.first_piola_kirchhoff_rate_tangent_stiffness(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
                    solver,
                    time,
                    DeformationGradient::identity(),
                    |t: Scalar| {
                        vector[0] = deformation_gradient_rate_11(t);
                        vector[4] = deformation_gradient_rate_22(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                )
            }
        } {
            Ok(results) => Ok(results),
            Err(error) => Err(ConstitutiveError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
