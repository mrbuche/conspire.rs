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
//! Consequently, the rate tangent stiffness associated with the first Piola-Kirchoff stress is symmetric for elastic-hyperviscous models.
//!
//! ```math
//! \mathcal{U}_{iJkL} = \mathcal{U}_{kLiJ}
//! ```

#[cfg(test)]
pub mod test;

mod almansi_hamel;

pub use almansi_hamel::AlmansiHamel;

use super::{super::fluid::viscous::Viscous, viscoelastic::Viscoelastic, *};
use crate::math::optimize::{NewtonRaphson, SecondOrder};
use std::fmt::Debug;

/// Required methods for elastic-hyperviscous constitutive models.
pub trait ElasticHyperviscous<'a>
where
    Self: Viscoelastic<'a> + Debug,
{
    /// Calculates and returns the dissipation potential.
    ///
    /// ```math
    /// \mathbf{P}^e(\mathbf{F}):\dot{\mathbf{F}} + \phi(\mathbf{F},\dot{\mathbf{F}})
    /// ```
    fn calculate_dissipation_potential(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self
            .calculate_first_piola_kirchoff_stress(deformation_gradient, &ZERO_10)?
            .full_contraction(deformation_gradient_rate)
            + self
                .calculate_viscous_dissipation(deformation_gradient, deformation_gradient_rate)?)
    }
    /// Calculates and returns the viscous dissipation.
    ///
    /// ```math
    /// \phi = \phi(\mathbf{F},\dot{\mathbf{F}})
    /// ```
    fn calculate_viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Scalar, ConstitutiveError>;
    /// Solve for the unknown components of the Cauchy stress and deformation gradient under uniaxial stress.
    fn solve_uniaxial<const W: usize>(
        &self,
        deformation_gradient_rate_11: impl Fn(Scalar) -> Scalar,
        evaluation_times: [Scalar; W],
    ) -> Result<(DeformationGradients<W>, CauchyStresses<W>), ConstitutiveError> {
        let mut cauchy_stresses = CauchyStresses::zero();
        let mut deformation_gradients = DeformationGradients::identity();
        let time_steps = evaluation_times.windows(2).map(|time| time[1] - time[0]);
        for ((index, time_step), time) in time_steps.enumerate().zip(evaluation_times.into_iter()) {
            (deformation_gradients[index + 1], cauchy_stresses[index + 1]) = self
                .solve_uniaxial_inner(
                    &deformation_gradients[index],
                    deformation_gradient_rate_11(time),
                    time_step,
                )?;
        }
        Ok((deformation_gradients, cauchy_stresses))
    }
    #[doc(hidden)]
    fn solve_uniaxial_inner(
        &self,
        deformation_gradient_previous: &DeformationGradient,
        deformation_gradient_rate_11: Scalar,
        time_step: Scalar,
    ) -> Result<(DeformationGradient, CauchyStress), ConstitutiveError> {
        let optimization = NewtonRaphson {
            ..Default::default()
        };
        let deformation_gradient = optimization.minimize(
            |deformation_gradient: &DeformationGradient| {
                let (deformation_gradient_rate, _) = self.solve_uniaxial_inner_inner(
                    deformation_gradient,
                    &deformation_gradient_rate_11,
                )?;
                Ok(deformation_gradient.copy()
                    - deformation_gradient_previous
                    - &deformation_gradient_rate * time_step)
            },
            |deformation_gradient: &DeformationGradient| {
                let (deformation_gradient_rate, _) = self.solve_uniaxial_inner_inner(
                    deformation_gradient,
                    &deformation_gradient_rate_11,
                )?;
                Ok(IDENTITY_1010
                    - TensorRank4::dyad_ik_jl(
                        &(&deformation_gradient_rate * deformation_gradient.inverse()),
                        &IDENTITY_00,
                    ) * time_step)
            },
            IDENTITY_10,
            None,
            None,
        )?;
        let (_, cauchy_stress) =
            self.solve_uniaxial_inner_inner(&deformation_gradient, &deformation_gradient_rate_11)?;
        Ok((deformation_gradient, cauchy_stress))
    }
    #[doc(hidden)]
    fn solve_uniaxial_inner_inner(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate_11: &Scalar,
    ) -> Result<(DeformationGradientRate, CauchyStress), ConstitutiveError> {
        let optimization = NewtonRaphson {
            ..Default::default()
        };
        let deformation_gradient_rate_33 = optimization.minimize(
            |deformation_gradient_rate_33: &Scalar| {
                Ok(self.calculate_cauchy_stress(
                    deformation_gradient,
                    &DeformationGradientRate::new([
                        [*deformation_gradient_rate_11, 0.0, 0.0],
                        [0.0, *deformation_gradient_rate_33, 0.0],
                        [0.0, 0.0, *deformation_gradient_rate_33],
                    ]),
                )?[2][2])
            },
            |deformation_gradient_rate_33: &Scalar| {
                Ok(self.calculate_cauchy_rate_tangent_stiffness(
                    deformation_gradient,
                    &DeformationGradientRate::new([
                        [*deformation_gradient_rate_11, 0.0, 0.0],
                        [0.0, *deformation_gradient_rate_33, 0.0],
                        [0.0, 0.0, *deformation_gradient_rate_33],
                    ]),
                )?[2][2][2][2])
            },
            -deformation_gradient_rate_11 / deformation_gradient[0][0].powf(1.5),
            None,
            None,
        )?;
        let deformation_gradient_rate = DeformationGradientRate::new([
            [*deformation_gradient_rate_11, 0.0, 0.0],
            [0.0, deformation_gradient_rate_33, 0.0],
            [0.0, 0.0, deformation_gradient_rate_33],
        ]);
        let cauchy_stress =
            self.calculate_cauchy_stress(deformation_gradient, &deformation_gradient_rate)?;
        Ok((deformation_gradient_rate, cauchy_stress))
    }
}
