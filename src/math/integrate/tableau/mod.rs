#[cfg(test)]
mod test;

use crate::math::Scalar;

/// Butcher tableau for an explicit Runge–Kutta method.
pub trait ButcherTableau {
    /// Number of stages, equal to the number of stored slopes.
    const STAGES: usize;
    /// Order of the propagating solution; the exponent for adaptive time steps.
    const ORDER: Scalar;
    /// Stage-coupling coefficients, `A[i].len() == i`.
    const A: &'static [&'static [Scalar]];
    /// Stage abscissae, `C[0] == 0`.
    const C: &'static [Scalar];
    /// Propagating weights.
    const B: &'static [Scalar];
    /// Whether the last stage of a step is the first stage of the next — the
    /// last row of `A` equals `B` and `C` ends at 1. Lets a stepping loop reuse
    /// the final rate evaluation as the next step's first.
    const FSAL: bool = false;
}

/// An embedded explicit Runge–Kutta pair.
pub trait EmbeddedTableau: ButcherTableau {
    /// Difference of the propagating and embedded weights.
    const D: &'static [Scalar];
}
