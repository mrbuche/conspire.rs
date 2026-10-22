#[cfg(test)]
mod test;

use super::{Damage, HyperDamage};
use crate::{
    constitutive::{
        Constitutive, ConstitutiveError,
        cohesive::{Cohesive, Stiffnesses, SymmetricStiffnesses, Tractions},
    },
    math::Quantity,
    mechanics::Scalar,
    units::{ForcePerLength, Length, Stress},
};

/// The trapezoidal Tvergaard-Hutchinson cohesive constitutive model.
///
/// The effective opening is $`\lambda=\sqrt{\langle\delta_n\rangle^2/\delta_n^{*2}+\delta_t^2/\delta_t^{*2}}`$,
/// so compression does not open the interface, and the tractions are
/// $`T_n=\sigma_\mathrm{max}\,g\,\delta_n/\delta_n^*`$ and
/// $`T_t=\sigma_\mathrm{max}\,(\delta_n^*/\delta_t^*)\,g\,\delta_t/\delta_t^*`$ for $`\delta_n\ge0`$,
/// where $`g=F(\lambda_\mathrm{max})/\lambda_\mathrm{max}`$ and
/// $`F`$ rises linearly to one at $`\lambda_1`$, stays there to $`\lambda_2`$, and falls linearly to zero at one.
/// Compression has the same secant stiffness as tension, which vanishes once the interface has failed.
///
/// The tractions derive from the potential
/// $`\Phi=\sigma_\mathrm{max}\delta_n^*\big(\int_0^{\lambda_\mathrm{max}}F-\tfrac{1}{2}g\lambda_\mathrm{max}^2+\tfrac{1}{2}g(\delta_n^2/\delta_n^{*2}+\delta_t^2/\delta_t^{*2})\big)`$,
/// so the stiffness is symmetric.
/// Because the secant stiffness in compression depends on the opening,
/// the shear traction while softening in compression is
/// $`T_t=\sigma_\mathrm{max}(\delta_n^*/\delta_t^*)\big(g\,\delta_t/\delta_t^*+\tfrac{1}{2}g'\,\delta_n^2\delta_t/(\delta_n^{*2}\delta_t^*\lambda)\big)`$.
/// The energy to fail the interface is $`\tfrac{1}{2}\sigma_\mathrm{max}\delta_n^*(1-\lambda_1+\lambda_2)`$.
#[derive(Clone, Debug)]
pub struct TvergaardHutchinson {
    /// The strength $`\sigma_\mathrm{max}`$.
    pub strength: Quantity<Stress>,
    /// The normal length $`\delta_n^*`$.
    pub normal_length: Quantity<Length>,
    /// The tangential length $`\delta_t^*`$.
    pub tangential_length: Quantity<Length>,
    /// The opening $`\lambda_1`$ the strength is reached at.
    pub onset: Scalar,
    /// The opening $`\lambda_2`$ the strength starts to fall at.
    pub softening: Scalar,
}

impl Constitutive for TvergaardHutchinson {}

impl Cohesive for TvergaardHutchinson {}

impl TvergaardHutchinson {
    fn check(&self) -> Result<(), ConstitutiveError> {
        if 0.0 < self.onset && self.onset <= self.softening && self.softening < 1.0 {
            Ok(())
        } else {
            Err(ConstitutiveError::custom(
                "The openings must satisfy 0 < onset <= softening < 1.",
                self,
            ))
        }
    }
    fn scaled(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
    ) -> (Scalar, Scalar) {
        (
            normal_separation.ratio(self.normal_length).value(),
            tangential_separation.ratio(self.tangential_length).value(),
        )
    }
    fn tangential_strength(&self) -> Quantity<Stress> {
        self.strength * self.normal_length.ratio(self.tangential_length)
    }
    fn secant(&self, lambda: Scalar) -> (Scalar, Scalar, Scalar) {
        if lambda < self.onset {
            (1.0 / self.onset, 0.0, 0.0)
        } else if lambda < self.softening {
            (1.0 / lambda, -1.0 / lambda.powi(2), 2.0 / lambda.powi(3))
        } else if lambda < 1.0 {
            let slope = 1.0 / (1.0 - self.softening);
            (
                (1.0 - lambda) * slope / lambda,
                -slope / lambda.powi(2),
                2.0 * slope / lambda.powi(3),
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    }
    fn area(&self, lambda: Scalar) -> Scalar {
        if lambda < self.onset {
            lambda.powi(2) / (2.0 * self.onset)
        } else if lambda < self.softening {
            lambda - 0.5 * self.onset
        } else if lambda < 1.0 {
            let rest = 1.0 - self.softening;
            self.softening - 0.5 * self.onset + 0.5 * (rest - (1.0 - lambda).powi(2) / rest)
        } else {
            0.5 * (1.0 + self.softening - self.onset)
        }
    }
}

impl Damage for TvergaardHutchinson {
    fn tractions(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
        state: Quantity,
    ) -> Result<Tractions, ConstitutiveError> {
        self.check()?;
        let (a, b) = self.scaled(normal_separation, tangential_separation);
        let opening = a.max(0.0).hypot(b);
        let (g, slope, _) = self.secant(opening.max(state.value()));
        let compressive = if a < 0.0 && opening > 0.0 && opening >= state.value() {
            0.5 * slope * a * a * b / opening
        } else {
            0.0
        };
        Ok((
            self.strength * (g * a),
            self.tangential_strength() * (g * b + compressive),
        ))
    }
    fn stiffnesses(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
        state: Quantity,
    ) -> Result<Stiffnesses, ConstitutiveError> {
        let (normal, coupling, tangential) =
            self.symmetric_stiffnesses(normal_separation, tangential_separation, state)?;
        Ok((normal, coupling, coupling, tangential))
    }
    fn opening(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
    ) -> Result<Quantity, ConstitutiveError> {
        let (a, b) = self.scaled(normal_separation, tangential_separation);
        Ok(Quantity::new(a.max(0.0).hypot(b)))
    }
}

impl HyperDamage for TvergaardHutchinson {
    fn potential(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
        state: Quantity,
    ) -> Result<Quantity<ForcePerLength>, ConstitutiveError> {
        self.check()?;
        let (a, b) = self.scaled(normal_separation, tangential_separation);
        let lambda = a.max(0.0).hypot(b).max(state.value());
        let (g, _, _) = self.secant(lambda);
        Ok(self.strength
            * self.normal_length
            * (self.area(lambda) - 0.5 * g * lambda.powi(2) + 0.5 * g * (a * a + b * b)))
    }
    fn symmetric_stiffnesses(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
        state: Quantity,
    ) -> Result<SymmetricStiffnesses, ConstitutiveError> {
        self.check()?;
        let (a, b) = self.scaled(normal_separation, tangential_separation);
        let opening = a.max(0.0).hypot(b);
        let loading = opening >= state.value();
        let (g, slope, curvature) = self.secant(opening.max(state.value()));
        let (slope, curvature) = if loading {
            (slope, curvature)
        } else {
            (0.0, 0.0)
        };
        let ratio = if opening > 0.0 { slope / opening } else { 0.0 };
        let compressive = a < 0.0;
        let normal = g + if compressive { 0.0 } else { ratio * a * a };
        let tangential = g
            + ratio * b * b
            + if compressive {
                0.5 * curvature * a * a
            } else {
                0.0
            };
        Ok((
            self.strength / self.normal_length * normal,
            self.strength / self.tangential_length * (ratio * a * b),
            self.tangential_strength() / self.tangential_length * tangential,
        ))
    }
}
