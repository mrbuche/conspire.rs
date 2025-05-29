//! Elastic-hyperviscous constitutive models.
//!
//! ---
//!
//! Elastic-hyperviscous constitutive models are defined by an elastic stress tensor function and a viscous dissipation function.
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

use super::{super::fluid::viscous::Viscous, viscoelastic::Viscoelastic, *};
use crate::math::{
    Matrix, TensorVec, Vector,
    optimize::{EqualityConstraint, FirstOrderRootFinding, NewtonRaphson, OptimizeError},
};
use std::fmt::Debug;

/// Required methods for elastic-hyperviscous constitutive models.
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
    /// Solve for the unknown components of the deformation gradient and rate under an applied load.
    ///
    /// ```math
    /// \mathbf{P}(\mathbf{F},\dot{\mathbf{F}}) - \boldsymbol{\lambda} - \mathbf{P}_0 = \mathbf{0}
    /// ```
    fn root_uniaxial<const W: usize>(
        &self,
        deformation_gradient_rate_11: impl Fn(Scalar) -> Scalar,
        evaluation_times: [Scalar; W],
    ) -> Result<(DeformationGradients<W>, DeformationGradientRates<W>), ConstitutiveError> {
        let mut deformation_gradients = DeformationGradients::identity();
        let mut deformation_gradient_rates = DeformationGradientRates::zero();
        let time_steps = evaluation_times.windows(2).map(|time| time[1] - time[0]);
        for ((index, time_step), time) in time_steps.enumerate().zip(evaluation_times.into_iter()) {
            (
                deformation_gradients[index + 1],
                deformation_gradient_rates[index + 1],
            ) = self.root_uniaxial_inner(
                &deformation_gradients[index],
                deformation_gradient_rate_11(time),
                time_step,
            )?;
        }
        Ok((deformation_gradients, deformation_gradient_rates))
    }
    #[doc(hidden)]
    fn root_uniaxial_inner(
        &self,
        deformation_gradient_previous: &DeformationGradient,
        deformation_gradient_rate_11: Scalar,
        time_step: Scalar,
    ) -> Result<(DeformationGradient, DeformationGradientRate), OptimizeError> {
        let solver = NewtonRaphson {
            ..Default::default()
        };
        let deformation_gradient = solver.root(
            |deformation_gradient: &DeformationGradient| {
                Ok(deformation_gradient.clone()
                    - deformation_gradient_previous
                    - &self.root_uniaxial_inner_inner(
                        deformation_gradient,
                        &deformation_gradient_rate_11,
                    )? * time_step)
            },
            |deformation_gradient: &DeformationGradient| {
                Ok(IDENTITY_1010
                    - TensorRank4::dyad_ik_jl(
                        &(&self.root_uniaxial_inner_inner(
                            deformation_gradient,
                            &deformation_gradient_rate_11,
                        )? * deformation_gradient.inverse()),
                        &IDENTITY_00,
                    ) * time_step)
            },
            IDENTITY_10,
            EqualityConstraint::None,
        )?;
        let deformation_gradient_rate =
            self.root_uniaxial_inner_inner(&deformation_gradient, &deformation_gradient_rate_11)?;
        Ok((deformation_gradient, deformation_gradient_rate))
    }
    #[doc(hidden)]
    fn root_uniaxial_inner_inner(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate_11: &Scalar,
    ) -> Result<DeformationGradientRate, OptimizeError> {
        let solver = NewtonRaphson {
            ..Default::default()
        };
        let mut matrix = Matrix::zero(4, 9);
        let mut vector = Vector::zero(4);
        matrix[0][0] = 1.0;
        matrix[1][1] = 1.0;
        matrix[2][2] = 1.0;
        matrix[3][5] = 1.0;
        vector[0] = *deformation_gradient_rate_11;
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
            DeformationGradientRate::zero(),
            EqualityConstraint::Linear(matrix, vector),
        )
    }
}
