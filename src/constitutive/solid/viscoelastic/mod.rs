//! Viscoelastic solid constitutive models.
//!
//! ---
//!
#![doc = include_str!("doc.md")]

#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
pub mod test;

mod canonical;

use super::{super::fluid::viscous::Viscous, *};
use crate::{
    math::{
        Matrix, Quantity, Vector,
        integrate::ImplicitDaeRoot,
        optimize::{EqualityConstraint, RootFinding},
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
    /// Calculates and returns the tangent stiffness associated with the Cauchy stress.
    ///
    /// ```math
    /// \mathcal{T}_{ijkL} = \frac{\partial\sigma_{ij}}{\partial F_{kL}} = J^{-1} \mathcal{G}_{MNkL} F_{iM} F_{jN} - \sigma_{ij} F_{kL}^{-T} + \left(\delta_{jk}\sigma_{is} + \delta_{ik}\sigma_{js}\right)F_{sL}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let cauchy_stress = self.cauchy_stress(deformation_gradient, deformation_gradient_rate)?;
        let some_stress = &cauchy_stress * &deformation_gradient_inverse_transpose;
        Ok(self
            .second_piola_kirchhoff_tangent_stiffness(
                deformation_gradient,
                deformation_gradient_rate,
            )?
            .contract_first_second_with_second(deformation_gradient, deformation_gradient)
            / deformation_gradient.determinant()
            - CauchyTangentStiffness::dyad_ij_kl(
                &cauchy_stress,
                &deformation_gradient_inverse_transpose,
            )
            + CauchyTangentStiffness::dyad_il_kj(&some_stress, &IDENTITY)
            + CauchyTangentStiffness::dyad_ik_jl(&IDENTITY, &some_stress))
    }
    /// Calculates and returns the tangent stiffness associated with the first Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{C}_{iJkL} = \frac{\partial P_{iJ}}{\partial F_{kL}} = J \mathcal{T}_{iskL} F_{sJ}^{-T} + P_{iJ} F_{kL}^{-T} - P_{iL} F_{kJ}^{-T}
    /// ```
    fn first_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let first_piola_kirchhoff_stress =
            self.first_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_rate)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_rate)?
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
    /// Calculates and returns the tangent stiffness associated with the second Piola-Kirchhoff stress.
    ///
    /// ```math
    /// \mathcal{G}_{IJkL} = \frac{\partial S_{IJ}}{\partial F_{kL}} = \mathcal{C}_{mJkL}F_{mI}^{-T} - S_{LJ}F_{kI}^{-T} = J \mathcal{T}_{mnkL} F_{mI}^{-T} F_{nJ}^{-T} + S_{IJ} F_{kL}^{-T} - S_{IL} F_{kJ}^{-T} -S_{LJ} F_{kI}^{-T}
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let deformation_gradient_inverse_transpose = deformation_gradient.inverse_transpose();
        let deformation_gradient_inverse = deformation_gradient_inverse_transpose.transpose();
        let second_piola_kirchhoff_stress =
            self.second_piola_kirchhoff_stress(deformation_gradient, deformation_gradient_rate)?;
        Ok(self
            .cauchy_tangent_stiffness(deformation_gradient, deformation_gradient_rate)?
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

/// Root-finding methods for viscoelastic solid constitutive models.
pub trait Root {
    /// Solve for the unknown components of the deformation gradient and rate under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\dot{\mathbf{F}}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeRoot<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradient,
            DeformationGradients,
            DeformationGradientRates,
        >,
        solver: impl RootFinding<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradientRate,
        >,
    ) -> Result<(Times, DeformationGradients, DeformationGradientRates), ConstitutiveError>;
}

impl<T> Root for T
where
    T: Viscoelastic,
{
    fn root(
        &self,
        applied_load: AppliedLoad,
        integrator: impl ImplicitDaeRoot<
            FirstPiolaKirchhoffStress,
            FirstPiolaKirchhoffRateTangentStiffness,
            DeformationGradient,
            DeformationGradients,
            DeformationGradientRates,
        >,
        solver: impl RootFinding<
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
                        vec![EqualityConstraint::Linear(matrix.clone(), vector.clone())]
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
                        vec![EqualityConstraint::Linear(matrix.clone(), vector.clone())]
                    },
                )
            }
        }
        .map_err(|error| ConstitutiveError::upstream(error, self))
    }
}
