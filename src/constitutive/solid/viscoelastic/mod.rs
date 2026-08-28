//! Viscoelastic solid constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
pub mod test;

use super::{super::fluid::viscous::Viscous, *};
use crate::{
    math::{
        Matrix, Quantity, Vector,
        integrate::{ImplicitDaeFirstOrderRoot, ImplicitDaeZerothOrderRoot},
        optimize::{EqualityConstraint, FirstOrderRootFinding, ZerothOrderRootFinding},
    },
    units::Time,
};

/// Possible applied loads.
pub enum AppliedLoad<'a> {
    /// Uniaxial stress given $`\dot{F}_{11}`$.
    UniaxialStress(fn(Quantity<Time>) -> Scalar, &'a [Quantity<Time>]),
    /// Biaxial stress given $`\dot{F}_{11}`$ and $`\dot{F}_{22}`$.
    BiaxialStress(
        fn(Quantity<Time>) -> Scalar,
        fn(Quantity<Time>) -> Scalar,
        &'a [Quantity<Time>],
    ),
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
            .contract_first_second_with_second(deformation_gradient, deformation_gradient)
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
            .contract_second_with_first(&deformation_gradient.inverse_transpose())
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
            .contract_first_second_with_second(
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
        integrator: impl ImplicitDaeZerothOrderRoot<
            FirstPiolaKirchhoffStress,
            DeformationGradient,
            DeformationGradients,
            DeformationGradientRates,
        >,
        solver: impl ZerothOrderRootFinding<FirstPiolaKirchhoffStress, DeformationGradientRate>,
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
            DeformationGradient,
            DeformationGradients,
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
        integrator: impl ImplicitDaeZerothOrderRoot<
            FirstPiolaKirchhoffStress,
            DeformationGradient,
            DeformationGradients,
            DeformationGradientRates,
        >,
        solver: impl ZerothOrderRootFinding<FirstPiolaKirchhoffStress, DeformationGradientRate>,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), ConstitutiveError> {
        match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_rate_11, time) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                integrator.integrate(
                    |_: Quantity<Time>,
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
                    |t: Quantity<Time>| {
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
                    |_: Quantity<Time>,
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
                    |t: Quantity<Time>| {
                        vector[0] = deformation_gradient_rate_11(t);
                        vector[4] = deformation_gradient_rate_22(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                )
            }
        }
        .map_err(|error| ConstitutiveError::upstream(error, self))
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
            DeformationGradient,
            DeformationGradients,
            DeformationGradientRates,
        >,
        solver: impl FirstOrderRootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradientRate,
        >,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), ConstitutiveError> {
        match applied_load {
            AppliedLoad::UniaxialStress(deformation_gradient_rate_11, time) => {
                let mut matrix = Matrix::zero(4, 9);
                let mut vector = Vector::zero(4);
                matrix[0][0] = 1.0;
                matrix[1][1] = 1.0;
                matrix[2][2] = 1.0;
                matrix[3][5] = 1.0;
                integrator.integrate(
                    |_: Quantity<Time>,
                     deformation_gradient: &DeformationGradient,
                     deformation_gradient_rate: &DeformationGradientRate| {
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
                    |_: Quantity<Time>,
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
                    |t: Quantity<Time>| {
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
                    |_: Quantity<Time>,
                     deformation_gradient: &DeformationGradient,
                     deformation_gradient_rate: &DeformationGradientRate| {
                        Ok(self.first_piola_kirchhoff_stress(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )?)
                    },
                    |_: Quantity<Time>,
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
                    |t: Quantity<Time>| {
                        vector[0] = deformation_gradient_rate_11(t);
                        vector[4] = deformation_gradient_rate_22(t);
                        EqualityConstraint::Linear(matrix.clone(), vector.clone())
                    },
                )
            }
        }
        .map_err(|error| ConstitutiveError::upstream(error, self))
    }
}
