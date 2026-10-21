#[cfg(test)]
mod test;

use crate::{
    math::{Quantity, Scalar},
    units::UnitRatio,
};

/// The step-size law of an adaptive integrator.
///
/// ```math
/// h_{n+1} = h_n\,\mathrm{clamp}\left[\beta\left(\frac{e_\mathrm{tol}}{e_{n+1}}\right)^{1/p},\ c,\ g\right]
/// ```
///
/// The step grows by the ceiling $`g`$ when the error vanishes. The same law serves
/// every adaptive loop, whatever the test that accepts a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepController {
    /// The multiplier $`\beta`$.
    pub beta: Scalar,
    /// The order $`p`$ of the error estimate.
    pub order: Scalar,
    /// The cut back factor $`c`$, the floor of the scale.
    pub cut: Scalar,
    /// The growth factor $`g`$, the ceiling of the scale.
    pub grow: Scalar,
}

impl StepController {
    pub const fn new(beta: Scalar, order: Scalar, cut: Scalar, grow: Scalar) -> Self {
        Self {
            beta,
            order,
            cut,
            grow,
        }
    }
    /// The factor to multiply the step by given the error and the tolerance it is held to.
    pub fn scale<U>(&self, error: Quantity<U>, tolerance: Quantity<U>) -> Scalar
    where
        U: UnitRatio,
    {
        if error > Quantity::default() {
            (self.beta * (tolerance / error).value().powf(1.0 / self.order))
                .clamp(self.cut, self.grow)
        } else {
            self.grow
        }
    }
}
