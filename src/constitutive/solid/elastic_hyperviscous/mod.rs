//! Elastic-hyperviscous solid constitutive models.
//!
//! ---
//!
//! Elastic-hyperviscous solid constitutive models are defined by an elastic stress tensor function and a viscous dissipation function.
//!
//! ```math
//! \mathbf{P}:\dot{\mathbf{F}} - \mathbf{P}^e(\mathbf{F}):\dot{\mathbf{F}} - \phi(\mathbf{F},\dot{\mathbf{F}}) \geq 0
//! ```
//! Satisfying the second law of thermodynamics though a minimum viscous dissipation principal yields a relation for the stress.
//!
//! ```math
//! \mathbf{P} = \mathbf{P}^e + \frac{\partial\phi}{\partial\dot{\mathbf{F}}}
//! ```
//! Consequently, the rate tangent stiffness associated with the first Piola-Kirchhoff stress is symmetric for these constitutive models.
//!
//! ```math
//! \mathcal{U}_{iJkL} = \mathcal{U}_{kLiJ}
//! ```

#[cfg(test)]
pub mod test;

mod almansi_hamel;

pub use almansi_hamel::AlmansiHamel;

use super::{
    viscoelastic::{AppliedLoad, Viscoelastic},
    *,
};
use crate::math::{
    Matrix, Vector,
    integrate::{ImplicitDaeFirstOrderMinimize, ImplicitDaeSecondOrderMinimize},
    optimize::{EqualityConstraint, FirstOrderOptimization, SecondOrderOptimization},
};

/// Required methods for elastic-hyperviscous solid constitutive models.
pub trait ElasticHyperviscous
where
    Self: Viscoelastic,
{
    /// Calculates and returns the dissipation potential.
    ///
    /// ```math
    /// \mathbf{P}^e(\mathbf{F}):\dot{\mathbf{F}} + \phi(\mathbf{F},\dot{\mathbf{F}})
    /// ```
    fn dissipation_potential(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self
            .first_piola_kirchhoff_stress(deformation_gradient, &ZERO_10)?
            .full_contraction(deformation_gradient_rate)
            + self.viscous_dissipation(deformation_gradient, deformation_gradient_rate)?)
    }
    /// Calculates and returns the viscous dissipation.
    ///
    /// ```math
    /// \phi = \phi(\mathbf{F},\dot{\mathbf{F}})
    /// ```
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Scalar, ConstitutiveError>;
}

/// First-order optimization methods for elastic-hyperviscous solid constitutive models.
pub trait FirstOrderMinimize {
    /// Solve for the unknown components of the deformation gradient and rate under an applied load.
    ///
    /// ```math
    /// \Pi(\mathbf{F},\dot{\mathbf{F}},\boldsymbol{\lambda}) = \mathbf{P}^e(\mathbf{F}):\dot{\mathbf{F}} + \phi(\mathbf{F},\dot{\mathbf{F}}) - \boldsymbol{\lambda}:(\dot{\mathbf{F}} - \dot{\mathbf{F}}_0) - \mathbf{P}_0:\dot{\mathbf{F}}
    /// ```
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeFirstOrderMinimize<
            Scalar,
            DeformationGradientRate,
            DeformationGradientRates,
        >,
        solver: impl FirstOrderOptimization<Scalar, DeformationGradient>,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), ConstitutiveError>;
}

/// Second-order optimization methods for elastic-hyperviscous solid constitutive models.
pub trait SecondOrderMinimize {
    /// Solve for the unknown components of the deformation gradient and rate under an applied load.
    ///
    /// ```math
    /// \Pi(\mathbf{F},\dot{\mathbf{F}},\boldsymbol{\lambda}) = \mathbf{P}^e(\mathbf{F}):\dot{\mathbf{F}} + \phi(\mathbf{F},\dot{\mathbf{F}}) - \boldsymbol{\lambda}:(\dot{\mathbf{F}} - \dot{\mathbf{F}}_0) - \mathbf{P}_0:\dot{\mathbf{F}}
    /// ```
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeSecondOrderMinimize<
            Scalar,
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradientRate,
            DeformationGradientRates,
        >,
        solver: impl SecondOrderOptimization<
            Scalar,
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradient,
        >,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), ConstitutiveError>;
}

impl<T> FirstOrderMinimize for T
where
    T: ElasticHyperviscous,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeFirstOrderMinimize<
            Scalar,
            DeformationGradientRate,
            DeformationGradientRates,
        >,
        solver: impl FirstOrderOptimization<Scalar, DeformationGradientRate>,
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
                        Ok(self.dissipation_potential(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
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
                    (
                        DeformationGradient::identity(),
                        DeformationGradientRate::zero(),
                    ),
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
                        Ok(self.dissipation_potential(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
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
                    (
                        DeformationGradient::identity(),
                        DeformationGradientRate::zero(),
                    ),
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

impl<T> SecondOrderMinimize for T
where
    T: ElasticHyperviscous,
{
    fn minimize(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeSecondOrderMinimize<
            Scalar,
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradientRate,
            DeformationGradientRates,
        >,
        solver: impl SecondOrderOptimization<
            Scalar,
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
                        Ok(self.dissipation_potential(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
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
                    (
                        DeformationGradient::identity(),
                        DeformationGradientRate::zero(),
                    ),
                    |t: Scalar| {
                        vector[0] = deformation_gradient_rate_11(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                    None,
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
                        Ok(self.dissipation_potential(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
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
                    (
                        DeformationGradient::identity(),
                        DeformationGradientRate::zero(),
                    ),
                    |t: Scalar| {
                        vector[0] = deformation_gradient_rate_11(t);
                        vector[4] = deformation_gradient_rate_22(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                    None,
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
