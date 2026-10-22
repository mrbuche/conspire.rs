#[cfg(test)]
mod test;

use super::{Damage, HyperDamage};
use crate::{
    constitutive::{
        Constitutive, ConstitutiveError,
        cohesive::{Cohesive, Stiffnesses, Tractions},
    },
    math::{Quantity, integrate::StepError},
    mechanics::Scalar,
    units::{ForcePerLength, Length, Stress},
};

const ONSET_TOLERANCE: Scalar = 1e-6;

#[doc = include_str!("doc.md")]
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
    #[doc = include_str!("tractions.md")]
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
    #[doc = include_str!("stiffnesses.md")]
    fn stiffnesses(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
        state: Quantity,
    ) -> Result<Stiffnesses, ConstitutiveError> {
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
        let coupling = ratio * a * b;
        Ok((
            self.strength / self.normal_length * normal,
            self.strength / self.tangential_length * coupling,
            self.tangential_strength() / self.normal_length * coupling,
            self.tangential_strength() / self.tangential_length * tangential,
        ))
    }
    #[doc = include_str!("step_error.md")]
    fn step_error(
        &self,
        separation_old: (Quantity<Length>, Quantity<Length>),
        separation: (Quantity<Length>, Quantity<Length>),
        state_old: Quantity,
        state: Quantity,
    ) -> Result<StepError, ConstitutiveError> {
        self.check()?;
        let (a_old, b_old) = self.scaled(separation_old.0, separation_old.1);
        let (a, b) = self.scaled(separation.0, separation.1);
        let (a_old, a) = (a_old.max(0.0), a.max(0.0));
        let (opening_old, opening) = (a_old.hypot(b_old), a.hypot(b));
        let (state_old, state) = (state_old.value(), state.value());
        let increment = state - state_old;
        if increment <= 0.0 || opening <= 0.0 {
            return Ok(StepError::default());
        }
        let direction = (a / opening, b / opening);
        let direction_old = if opening_old > 0.0 {
            (a_old / opening_old, b_old / opening_old)
        } else {
            direction
        };
        let turning = (direction.0 - direction_old.0).hypot(direction.1 - direction_old.1);
        let reloaded = opening_old < state_old * (1.0 - ONSET_TOLERANCE) && state_old < opening;
        let overshoot = [self.onset, self.softening, 1.0]
            .into_iter()
            .filter(|&kink| state_old < kink && kink <= state)
            .map(|kink| state - kink)
            .chain(reloaded.then_some(opening - state_old))
            .fold(0.0, Scalar::max);
        Ok(StepError {
            direction: Quantity::new(0.5 * increment * turning),
            overshoot: Quantity::new(overshoot),
        })
    }
    #[doc = include_str!("opening.md")]
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
    #[doc = include_str!("potential.md")]
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
}
